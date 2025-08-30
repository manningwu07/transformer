package main

import (
	"bufio"
	"errors"
	"fmt"
	"os"
	"path/filepath"
	"regexp"
	"sort"
	"strings"

	"gonum.org/v1/gonum/mat"
)

// TrainingRecord is a generic dataset example (inputs column vector,
// targets column vector, both mat.Dense wrappers are easy to create).
type TrainingRecord struct {
	Inputs  *mat.Dense
	Targets *mat.Dense
}

type Vocab struct {
	TokenToID map[string]int
	IDToToken []string
}

// Globals initialized on first loadTrainingSet call.
var (
	enVocab, zhVocab Vocab
	embEN, embZH     *mat.Dense // (dModel x |V|)
)

// Special tokens kept at the start of the vocab
var special = []string{"<pad>", "<bos>", "<eos>", "<unk>"}

// -------- Public functions required by your training loop --------

// loadTrainingSet builds:
//   - EN/ZH vocabs, each of size dModel (to match your block output size)
//   - EN and ZH embeddings (dModel x |V|)
//   - returns TrainingRecord slice where Inputs are embedded EN token vectors
//     (dModel x 1) and Targets are one-hot ZH token vectors (dModel x 1)
func loadTrainingSet(gpt Transformer) ([]TrainingRecord, error) {
	if len(gpt.blocks) == 0 || gpt.blocks[0].attn == nil {
		return nil, errors.New("model is not initialized")
	}
	dModel := gpt.blocks[0].attn.dModel

	paths, err := findParallelFiles()
	if err != nil {
		return nil, err
	}

	var enLines, zhLines []string

	enLines, err = readLines(paths.en, 0)
	if err != nil {
		return nil, err
	}
	zhLines, err = readLines(paths.zh, 0)
	if err != nil {
		return nil, err
	}

	enTokSeqs := make([][]string, len(enLines))
	zhTokSeqs := make([][]string, len(zhLines))
	n := min(len(zhTokSeqs), len(enTokSeqs))

	enTokensAll := []string{}
	zhTokensAll := []string{}
	for i := 0; i < n; i++ {
		enTokSeqs[i] = tokenizeEN(enLines[i])
		zhTokSeqs[i] = tokenizeZH(zhLines[i])
		enTokensAll = append(enTokensAll, enTokSeqs[i]...)
		zhTokensAll = append(zhTokensAll, zhTokSeqs[i]...)
	}

	// Build fixed-size vocabs of size dModel
	enVocab = buildVocab(enTokensAll, dModel)
	zhVocab = buildVocab(zhTokensAll, dModel)

	// Initialize embeddings (dModel x dModel here)
	embEN = initEmbeddings(dModel, enVocab)
	embZH = initEmbeddings(dModel, zhVocab)

	// Create position-aligned token pairs and convert to training samples
	pairs := alignByPosition(enTokSeqs, zhTokSeqs)
	records := make([]TrainingRecord, 0, len(pairs))
	for _, p := range pairs {
		enID := vocabLookup(enVocab, p[0])
		zhID := vocabLookup(zhVocab, p[1])

		// Input: embedded EN vector (d x 1)
		X := embed(embEN, enID)

		// Target: one-hot over size dModel, index = zhID
		T := oneHot(dModel, zhID)

		records = append(records, TrainingRecord{
			Inputs:  X,
			Targets: T,
		})
	}

	return records, nil
}

// evaluateAccuracy computes simple top-1 accuracy using a held-out slice of
// up to 10,000 records from the same building procedure as training.
// It forwards through the model blocks and compares argmax(logits) vs argmax(target).
func evaluateAccuracy(gpt Transformer) (int, int) {
	// Rebuild a fresh dataset and use a slice for "test".
	// In a real setup, you would separate train/test files.
	records, err := loadTrainingSet(gpt)
	if err != nil || len(records) == 0 {
		return 0, 0
	}

	// Use up to 10,000 records for evaluation
	N := min(len(records), 10000)

	correct := 0
	for i := 0; i < N; i++ {
		X := records[i].Inputs
		for l := 0; l < layers; l++ {
			X = gpt.blocks[l].Forward(X)
		}
		logits := UnembedZH(X)
		pred := argmaxVec(logits)
		tgt := argmaxVec(records[i].Targets)
		if pred == tgt {
			correct++
		}
	}
	return correct, N
}

// -------- Bilingual vocab and embeddings (EN-ZH) --------

// Build a vocab of fixed size 'size'. We reserve space for specials first,
// then fill with most frequent tokens from corpus.
func buildVocab(tokens []string, size int) Vocab {
	if size < len(special) {
		panic("vocab size must be >= number of special tokens")
	}
	// Count
	cnt := map[string]int{}
	for _, t := range tokens {
		if t == "" {
			continue
		}
		cnt[t]++
	}
	// Sort by frequency
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

	idToToken := make([]string, 0, size)
	idToToken = append(idToToken, special...)
	for _, p := range arr {
		if len(idToToken) >= size {
			break
		}
		// avoid duplicating specials if present
		skip := false
		for _, s := range special {
			if p.k == s {
				skip = true
				break
			}
		}
		if !skip {
			idToToken = append(idToToken, p.k)
		}
	}
	// If not enough unique tokens to fill the vocab, pad with dummy symbols
	for len(idToToken) < size {
		idToToken = append(idToToken, fmt.Sprintf("<pad%d>", len(idToToken)))
	}
	tokenToID := make(map[string]int, len(idToToken))
	for i, t := range idToToken {
		tokenToID[t] = i
	}
	return Vocab{TokenToID: tokenToID, IDToToken: idToToken}
}

func vocabLookup(v Vocab, tok string) int {
	if id, ok := v.TokenToID[tok]; ok {
		return id
	}
	return v.TokenToID["<unk>"]
}

// Initialize embeddings with small random values.
// Shape: (dModel x |V|)
func initEmbeddings(dModel int, v Vocab) *mat.Dense {
	data := randomArray(dModel*len(v.IDToToken), float64(dModel))
	return mat.NewDense(dModel, len(v.IDToToken), data)
}

// Column slice (copy) m[:, j] -> (r x 1)
func colAsVector(m *mat.Dense, j int) *mat.Dense {
	r, c := m.Dims()
	if j < 0 || j >= c {
		panic("colAsVector: column index out of range")
	}
	dst := make([]float64, r)
	for i := 0; i < r; i++ {
		dst[i] = m.At(i, j)
	}
	return mat.NewDense(r, 1, dst)
}

// Embed token id using emb matrix (returns dModel x 1)
func embed(emb *mat.Dense, id int) *mat.Dense {
	return colAsVector(emb, id)
}

// Unembed vector with tied weights (emb^T * x) -> (|V| x 1) logits
func unembed(emb *mat.Dense, x *mat.Dense) *mat.Dense {
	return dot(emb.T(), x).(*mat.Dense)
}

// -------- Tokenization (very lightweight) --------

// English: lowercase, split on letters+digits, keep apostrophes inside words
var enWordRE = regexp.MustCompile(`[a-z0-9]+(?:'[a-z0-9]+)?`)

func tokenizeEN(s string) []string {
	s = strings.ToLower(s)
	return enWordRE.FindAllString(s, -1)
}

// Chinese: split on runes, keeping only Han, punctuation/space removed.
// Latin sequences are grouped.
func tokenizeZH(s string) []string {
	out := []string{}
	var latin strings.Builder
	for _, r := range s {
		// Han range (CJK Unified Ideographs)
		if (r >= 0x4E00 && r <= 0x9FFF) ||
			(r >= 0x3400 && r <= 0x4DBF) {
			// flush latin if any
			if latin.Len() > 0 {
				out = append(out, strings.ToLower(latin.String()))
				latin.Reset()
			}
			out = append(out, string(r))
			continue
		}
		// Latin/number
		if (r >= 'A' && r <= 'Z') || (r >= 'a' && r <= 'z') ||
			(r >= '0' && r <= '9') || r == '\'' {
			latin.WriteRune(r)
			continue
		}
		// Whitespace or punctuation -> flush latin
		if latin.Len() > 0 {
			out = append(out, strings.ToLower(latin.String()))
			latin.Reset()
		}
	}
	if latin.Len() > 0 {
		out = append(out, strings.ToLower(latin.String()))
	}
	return out
}

// -------- Data loading (parallel EN-ZH) --------

type parallelPaths struct {
	en string
	zh string
}

// Try a few common locations for parallel files.
func findParallelFiles() (parallelPaths, error) {
	candidates := []parallelPaths{
		// Test subjects
		{en: "../data/test/train.test.eval.en", zh: "../data/test/train.test.eval.zh"},

		// Actual files I will pull from to train the modal
		// {en: "../data/tokenized/train.sample.en", zh: "../data/tokenized/train.sample.zh"},
		// Others are from
		// {en: "data/tokenized/train.en", zh: "data/tokenized/train.zh"},
		// {en: "data/tokenized/en.txt", zh: "data/tokenized/zh.txt"},
		// {en: "data/raw/wmt23-enzh/train.eng", zh: "data/raw/wmt23-enzh/train.zho"},
		// {en: "data/raw/wmt23-enzh/wmt23-enzh.en", zh: "data/raw/wmt23-enzh/wmt23-enzh.zh"},
	}
	for _, p := range candidates {
		if fileExists(p.en) && fileExists(p.zh) {
			return p, nil
		}
	}
	// As a last resort, scan the raw directory for *.en / *.zh pairs
	raw := "data/raw/wmt23-enzh"
	entries, err := os.ReadDir(raw)
	if err == nil {
		var ens, zhs []string
		for _, e := range entries {
			name := e.Name()
			if strings.HasSuffix(name, ".eng") {
				ens = append(ens, filepath.Join(raw, name))
			} else if strings.HasSuffix(name, ".zho") {
				zhs = append(zhs, filepath.Join(raw, name))
			}
		}
		if len(ens) > 0 && len(zhs) > 0 {
			return parallelPaths{en: ens[0], zh: zhs[0]}, nil
		}
	}
	return parallelPaths{}, errors.New("could not find parallel EN/ZH files")
}

func fileExists(p string) bool {
	_, err := os.Stat(p)
	return err == nil
}

func readLines(p string, limit int) ([]string, error) {
	f, err := os.Open(p)
	if err != nil {
		return nil, err
	}
	defer f.Close()
	out := []string{}
	sc := bufio.NewScanner(f)
	for sc.Scan() {
		out = append(out, sc.Text())
		if limit > 0 && len(out) >= limit {
			break
		}
	}
	return out, sc.Err()
}

// Until this gets fixed, accuracy will be shit
func alignByPosition(enTokens, zhTokens [][]string) [][2]string {
	pairs := make([][2]string, 0, 1024)
	for i := 0; i < len(enTokens) && i < len(zhTokens); i++ {
		a := enTokens[i]
		b := zhTokens[i]
		n := len(a)
		if len(b) < n {
			n = len(b)
		}
		for j := 0; j < n; j++ {
			if a[j] == "" || b[j] == "" {
				continue
			}
			pairs = append(pairs, [2]string{a[j], b[j]})
		}
	}
	return pairs
}

// save persists the whole Transformer using the gob-based SaveTransformer.
func save(gpt Transformer) error {
	_ = os.MkdirAll("models", 0o755)
	path := "models/transformer.gob"
	return SaveTransformer(&gpt, path)
}

// -------- Optional helpers to expose embedding/unembedding --------

// EmbedEN returns the (d x 1) embedding vector for an English token.
func EmbedEN(token string) *mat.Dense {
	if embEN == nil {
		panic("embeddings not initialized; call loadTrainingSet first")
	}
	id := vocabLookup(enVocab, token)
	return embed(embEN, id)
}

// UnembedZH returns (|Vzh| x 1) logits for a given (d x 1) hidden vector.
func UnembedZH(x *mat.Dense) *mat.Dense {
	if embZH == nil {
		panic("embeddings not initialized; call loadTrainingSet first")
	}
	return unembed(embZH, x)
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
