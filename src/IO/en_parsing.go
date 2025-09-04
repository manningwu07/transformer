package IO

import (
	"bufio"
	"errors"
	"fmt"
	"io"
	"os"
	"path/filepath"
	"sort"
	"strings"

	"gonum.org/v1/gonum/mat"

	"github.com/manningwu07/GPT/params"
	"github.com/manningwu07/GPT/transformer"
	"github.com/manningwu07/GPT/utils"
)

// TrainingRecord is a generic dataset example (inputs column vector,
// targets column vector, both mat.Dense wrappers are easy to create).
type TrainingRecord struct {
	Inputs  *mat.Dense
	Targets *mat.Dense
}

// Special tokens kept at the start of the vocab
var special = []string{"<pad>", "<bos>", "<eos>", "<unk>"}

// Build vocab  initialize embeddings from the training file in a streaming pass.
// Returns the number of lines seen (for logging) but does not load sequences into memory.
func BuildVocabAndEmbFromTrain(gpt transformer.Transformer) (int, error) {
	if len(gpt.Blocks) == 0 || gpt.Blocks[0].Attn == nil {
		return 0, errors.New("model is not initialized")
	}
	dModel := gpt.Blocks[0].Attn.DModel
	p := FindTrainFile()
	if p == "" {
		return 0, errors.New("could not find training file")
	}
	var err error
	params.Vocab, _, err = buildFixedVocabFromFile(p, params.Config.VocabSize)
	if err != nil {
		return 0, err
	}
	params.Emb = initEmbeddings(dModel, params.Vocab)

	if params.PosEmb == nil || params.PosEmb.RawMatrix().Rows != dModel || params.PosEmb.RawMatrix().Cols != params.Config.SeqLen {
		params.PosEmb = mat.NewDense(dModel, params.Config.SeqLen, utils.RandomArray(dModel*params.Config.SeqLen, float64(dModel)))
	}

	// return line count for logging
	lines, err := countLines(p)
	if err != nil {
		return 0, err
	}
	return lines, nil
}

// Add positional embeddings to an embedded sequence X (d x T), using positions 0..T-1.
func addPosToSequence(X *mat.Dense) *mat.Dense {
	if params.PosEmb == nil {
		return X
	}
	d, T := X.Dims()
	if T > params.PosEmb.RawMatrix().Cols {
		T = params.PosEmb.RawMatrix().Cols
	}
	out := mat.NewDense(d, T, nil)
	out.Copy(X)
	// out += posEmb[:, 0:T]
	for i := 0; i < d; i++ {
		for t := 0; t < T; t++ {
			out.Set(i, t, out.At(i, t)+params.PosEmb.At(i, t))
		}
	}
	return out
}

// Add positional embedding column posIdx to xCol (d x 1).
func AddPosCol(xCol *mat.Dense, posIdx int) *mat.Dense {
	if params.PosEmb == nil {
		return xCol
	}
	idx := posIdx
	if idx < 0 {
		idx = 0
	}
	if idx >= params.PosEmb.RawMatrix().Cols {
		idx = params.PosEmb.RawMatrix().Cols - 1
	}
	d, _ := xCol.Dims()
	out := mat.NewDense(d, 1, nil)
	for i := 0; i < d; i++ {
		out.Set(i, 0, xCol.At(i, 0)+params.PosEmb.At(i, idx))
	}
	return out
}

// load only token-id sequences for EVAL
func loadEvalSequences() ([][]int, error) {
	if len(params.Vocab.IDToToken) == 0 {
		return nil, errors.New("vocab not initialized; load train first")
	}
	p := findEvalFile()
	if p == "" {
		return nil, errors.New("could not find eval file")
	}
	lines, err := readLines(p, 0)
	if err != nil {
		return nil, err
	}
	seqIDs := make([][]int, len(lines))
	for i, s := range lines {
		toks := TokenizeENPieces(s) //ASCII Only
		toks = append([]string{"<bos>"}, toks...)
		toks = append(toks, "<eos>")
		ids := make([]int, len(toks))
		for j, t := range toks {
			ids[j] = VocabLookup(params.Vocab, t) // unseen -> <unk>
		}
		seqIDs[i] = ids
	}
	return seqIDs, nil
}

func FindTrainFile() string {
	if p := findEnglishFile(); p != "" {
		return p
	}
	return ""
}

func findEvalFile() string {
	candidates := []string{
		"../data/test/train.en",
		// "data/test/eval.en",
		// "data/raw/eval.eng",
	}
	for _, p := range candidates {
		if fileExists(p) {
			return p
		}
	}
	// fallback: first *.en named "eval" in tree
	root := "data"
	var first string
	_ = filepath.WalkDir(root, func(path string, d os.DirEntry, err error) error {
		if err == nil && !d.IsDir() && strings.HasSuffix(d.Name(), ".en") &&
			strings.Contains(strings.ToLower(d.Name()), "eval") {
			if first == "" {
				first = path
			}
		}
		return nil
	})
	return first
}

// uses eval.en and existing vocab/emb; streams examples.
func EvaluateMetrics(gpt transformer.Transformer) (int, int, float64) {
	seqs, err := loadEvalSequences()
	if err != nil || len(seqs) == 0 {
		return 0, 0, 0
	}
	limit := 10000
	total, correct := 0, 0
	ceSum := 0.0
	for _, ids := range seqs {
		if len(ids) < 2 {
			continue
		}
		// per-sequence KV caches, one per block
		type blkKV struct{ attnKV transformer.AttnKV }
		kvs := make([]blkKV, params.Layers)
		// roll through the sequence once; predict next token at each step
		// we run up to len(ids)-1 predictions
		var yLast *mat.Dense
		for t := 0; t+1 < len(ids); t++ {
			xLast := ColAsVector(params.Emb, ids[t])  // (dModel x 1)
			yLast = AddPosCol(xLast, kvs[0].attnKV.T) // add position for block 0 only
			for l := 0; l < params.Layers; l++ {
				yLast = gpt.Blocks[l].ForwardLastWithKV(yLast, &kvs[l].attnKV)
			}
			logits := Unembed(yLast) // emb^T * yLast
			pred := argmaxVec(logits)
			if pred == ids[t+1] {
				correct++
			}
			// accumulate CE
			oh := utils.OneHot(params.Config.VocabSize, ids[t+1])
			loss, _ := utils.CrossEntropyWithGrad(logits, oh)
			ceSum += loss

			total++
			if total >= limit {
				return correct, total, ceSum
			}
		}
	}
	return correct, total, ceSum
}

// loadTinyTrainIDs returns the first N training lines as BOS...EOS id sequences.
func LoadTinyTrainIDs(n int) ([][]int, error) {
	p := FindTrainFile()
	if p == "" {
		return nil, errors.New("could not find training file")
	}
	lines, err := readLines(p, n)
	if err != nil {
		return nil, err
	}
	out := make([][]int, 0, len(lines))
	for _, s := range lines {
		toks := TokenizeENPieces(s)
		if len(toks) == 0 {
			continue
		}
		ids := make([]int, 0, len(toks)+2)
		ids = append(ids, VocabLookup(params.Vocab, "<bos>"))
		for _, t := range toks {
			ids = append(ids, VocabLookup(params.Vocab, t))
		}
		ids = append(ids, VocabLookup(params.Vocab, "<eos>"))
		out = append(out, ids)
	}
	return out, nil
}

// Tokenization (ASCII-only pieces). Lowercases, drops any non 1-byte ASCII chars.
func TokenizeENPieces(s string) []string {
	// Fast ASCII lowercase non-ASCII drop
	b := make([]byte, 0, len(s))
	for i := 0; i < len(s); i++ {
		c := s[i]
		if c >= 'A' && c <= 'Z' {
			c = c + 32
		}
		if c < 0x80 {
			b = append(b, c)
		} else {
			// replace non-ascii with space to enforce splitting
			b = append(b, ' ')
		}
	}
	parts := strings.Fields(string(b))
	out := make([]string, 0, 16)
	for _, w := range parts {
		// split each ASCII word into 1–4 byte pieces
		for len(w) > 0 {
			take := min(4, len(w))
			out = append(out, w[:take])
			w = w[take:]
		}
	}
	return out
}

func buildFixedVocabFromCounts(cnt map[string]int, size int) params.Vocabulary {
	if size < len(special) {
		panic("vocab size must be >= number of special tokens")
	}
	type kv struct {
		k string
		v int
	}
	arr := make([]kv, 0, len(cnt))
	for k, v := range cnt {
		arr = append(arr, kv{k, v})
	}
	sort.Slice(arr, func(i, j int) bool {
		if arr[i].v == arr[j].v {
			return arr[i].k < arr[j].k
		}
		return arr[i].v > arr[j].v
	})
	idToToken := append([]string{}, special...)
	for _, p := range arr {
		if len(idToToken) >= size {
			break
		}
		skip := false
		for _, s := range special {
			if p.k == s {
				skip = true
				break
			}
		}
		if skip {
			continue
		}
		if !isASCIIString(p.k) {
			continue
		} // enforce 1-byte ASCII only
		if p.k == "" {
			continue
		}
		idToToken = append(idToToken, p.k)
	}
	for len(idToToken) < size {
		idToToken = append(idToToken, fmt.Sprintf("<pad%d>", len(idToToken)))
	}
	tok2id := map[string]int{}
	for i, t := range idToToken {
		tok2id[t] = i
	}
	return params.Vocabulary{TokenToID: tok2id, IDToToken: idToToken}
}

// Streaming vocab builder from file; returns vocab and number of lines processed.
func buildFixedVocabFromFile(path string, size int) (params.Vocabulary, int, error) {
	f, err := os.Open(path)
	if err != nil {
		return params.Vocabulary{}, 0, err
	}
	defer f.Close()
	r := bufio.NewReaderSize(f, 1<<20) // 1MB buffer
	counts := make(map[string]int, 1<<15)
	lines := 0
	for {
		line, err := r.ReadString('\n')
		if len(line) > 0 {
			lines++
			toks := TokenizeENPieces(line)
			for _, t := range toks {
				// enforce ASCII-only
				if t == "" || !isASCIIString(t) {
					continue
				}
				counts[t]++
			}
		}
		if err == io.EOF {
			break
		}
		if err != nil {
			return params.Vocabulary{}, lines, err
		}
	}
	return buildFixedVocabFromCounts(counts, size), lines, nil
}

// ASCII helper
func isASCIIString(s string) bool {
	for i := 0; i < len(s); i++ {
		if s[i] >= 0x80 {
			return false
		}
	}
	return true
}

func findEnglishFile() string {
	candidates := []string{
		"../data/test/train.en",
		// "../data/raw/train.eng",
	}
	for _, p := range candidates {
		if fileExists(p) {
			return p
		}
	}
	// fallback: first *.en in data tree
	fmt.Println("Failed to find candidate file, searching for first file in datatree.")
	root := "data"
	_ = filepath.WalkDir(root, func(path string, d os.DirEntry, err error) error {
		if err == nil && !d.IsDir() && strings.HasSuffix(d.Name(), ".en") {
			candidates = append(candidates, path)
		}
		return nil
	})
	if len(candidates) > 0 && fileExists(candidates[0]) {
		return candidates[0]
	}
	return ""
}

func VocabLookup(v params.Vocabulary, tok string) int {
	if id, ok := v.TokenToID[tok]; ok {
		return id
	}
	return v.TokenToID["<unk>"]
}

// Initialize embeddings with small random values.
// Shape: (dModel x |V|)
func initEmbeddings(dModel int, v params.Vocabulary) *mat.Dense {
	data := utils.RandomArray(dModel*len(v.IDToToken), float64(dModel))
	return mat.NewDense(dModel, len(v.IDToToken), data)
}

// Column slice (copy) m[:, j] -> (r x 1)
func ColAsVector(m *mat.Dense, j int) *mat.Dense {
	r, c := m.Dims()
	if j < 0 || j >= c {
		panic("colAsVector: column index out of range")
	}
	dst := make([]float64, r)
	for i := 0; i < r; i++ {
		dst[i] = m.At(i, j)
	}
	out := mat.NewDense(r, 1, dst)
	if params.PosEmb != nil {
		out = addPosToSequence(out)
	}
	return out
}

func EmbedSequence(emb *mat.Dense, ids []int) *mat.Dense {
	d, _ := emb.Dims()
	T := len(ids)
	out := mat.NewDense(d, T, nil)
	for t, id := range ids {
		for i := 0; i < d; i++ {
			out.Set(i, t, emb.At(i, id))
		}
	}
	return out
}

func Unembed(x *mat.Dense) *mat.Dense {
	if params.Emb == nil {
		panic("embedding not initialized; call loadTrainingSet first")
	}
	return utils.ToDense(utils.Dot(params.Emb.T(), x))
}

func fileExists(p string) bool {
	_, err := os.Stat(p)
	return err == nil
}

// readLines reads up to 'limit' lines (0 = no limit). Uses a large buffered reader.
func readLines(p string, limit int) ([]string, error) {
	f, err := os.Open(p)
	if err != nil {
		return nil, err
	}
	defer f.Close()
	r := bufio.NewReaderSize(f, 1<<20) // 1MB
	out := make([]string, 0, 4096)
	for {
		line, err := r.ReadString('\n')
		if len(line) > 0 {
			// trim trailing newline
			if line[len(line)-1] == '\n' {
				line = line[:len(line)-1]
			}
			out = append(out, line)
			if limit > 0 && len(out) >= limit {
				return out, nil
			}
		}
		if err == io.EOF {
			return out, nil
		}
		if err != nil {
			return out, err
		}
	}
}

// Stream training lines -> token IDs without loading all into memory.
type trainLineIter struct {
	path string
	f    *os.File
	r    *bufio.Reader
}

func NewTrainLineIter(path string) (*trainLineIter, error) {
	f, err := os.Open(path)
	if err != nil {
		return nil, err
	}
	return &trainLineIter{path: path, f: f, r: bufio.NewReaderSize(f, 1<<20)}, nil
}

func (it *trainLineIter) Close() error {
	if it.f != nil {
		return it.f.Close()
	}
	return nil
}

// nextIDs returns the next line converted to token IDs, or nil, io.EOF when at end.
// When EOF is reached, the iterator rewinds to the beginning to allow multiple epochs.
func (it *trainLineIter) NextIDs() ([]int, error) {
	for {
		line, err := it.r.ReadString('\n')
		if len(line) > 0 {
			toks := TokenizeENPieces(line)
			if len(toks) == 0 {
				// skip empty lines
				continue
			}
			// add BOS/EOS
			ids := make([]int, 0, len(toks)+2)
			ids = append(ids, VocabLookup(params.Vocab, "<bos>"))
			for _, t := range toks {
				ids = append(ids, VocabLookup(params.Vocab, t))
			}
			ids = append(ids, VocabLookup(params.Vocab, "<eos>"))
			return ids, nil
		}
		if err == io.EOF {
			// rewind for next epoch
			if _, err2 := it.f.Seek(0, io.SeekStart); err2 != nil {
				return nil, err2
			}
			it.r.Reset(it.f)
			return nil, io.EOF
		}
		if err != nil {
			return nil, err
		}
	}
}

// countLines returns number of lines in file (used for logging).
func countLines(path string) (int, error) {
	f, err := os.Open(path)
	if err != nil {
		return 0, err
	}
	defer f.Close()
	r := bufio.NewReaderSize(f, 1<<20)
	n := 0
	for {
		_, err := r.ReadString('\n')
		if err == io.EOF {
			return n, nil
		}
		if err != nil {
			return n, err
		}
		n++
	}
}

// -------- Small utilities --------

func argmaxVec(v *mat.Dense) int {
	r, c := v.Dims()
	if c != 1 {
		panic("argmaxVec expects a column vector")
	}
	bestI := 0
	best := v.At(0, 0)
	for i := 1; i < r; i++ {
		if v.At(i, 0) > best {
			best = v.At(i, 0)
			bestI = i
		}
	}
	return bestI
}
