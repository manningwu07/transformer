package main

import (
	"bufio"
	"errors"
	"fmt"
	"os"
	"path/filepath"
	"sort"
	"strings"
	"unicode/utf8"

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
	vocab Vocab
	emb   *mat.Dense // (dModel x |V|)
)

// Special tokens kept at the start of the vocab
var special = []string{"<pad>", "<bos>", "<eos>", "<unk>"}

// load only token-id sequences for TRAIN. Builds vocabemb once.
func loadTrainSequences(gpt Transformer) ([][]int, error) {
	if len(gpt.blocks) == 0 || gpt.blocks[0].attn == nil {
		return nil, errors.New("model is not initialized")
	}
	dModel := gpt.blocks[0].attn.dModel
	p := findTrainFile()
	if p == "" {
		return nil, errors.New("could not find training file")
	}
	lines, err := readLines(p, 0)
	if err != nil {
		return nil, err
	}
	// Build vocab from train only
	allTokens := []string{}
	seqTok := make([][]string, len(lines))
	for i, s := range lines {
		toks := tokenizeENPieces(s)
		toks = append([]string{"<bos>"}, toks...)
		toks = append(toks, "<eos>")
		seqTok[i] = toks
		allTokens = append(allTokens, toks...)
	}
	vocab = buildFixedVocab(allTokens, config.VocabSize)
	emb = initEmbeddings(dModel, vocab)
	// Convert to IDs
	seqIDs := make([][]int, len(seqTok))
	for i, toks := range seqTok {
		ids := make([]int, len(toks))
		for j, t := range toks {
			ids[j] = vocabLookup(vocab, t)
		}
		seqIDs[i] = ids
	}
	return seqIDs, nil
}

// load only token-id sequences for EVAL
func loadEvalSequences() ([][]int, error) {
	if len(vocab.IDToToken) == 0 {
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
		toks := tokenizeENPieces(s)
		toks = append([]string{"<bos>"}, toks...)
		toks = append(toks, "<eos>")
		ids := make([]int, len(toks))
		for j, t := range toks {
			ids[j] = vocabLookup(vocab, t) // unseen -> <unk>
		}
		seqIDs[i] = ids
	}
	return seqIDs, nil
}

func findTrainFile() string {
    if p := findEnglishFile(); p != "" { return p }
    return ""
}

func findEvalFile() string {
    candidates := []string{
        "../data/test/eval.en",
        "data/test/eval.en",
        "data/raw/eval.eng",
    }
    for _, p := range candidates {
        if fileExists(p) { return p }
    }
    // fallback: first *.en named "eval" in tree
    root := "data"
    var first string
    _ = filepath.WalkDir(root, func(path string, d os.DirEntry, err error) error {
        if err == nil && !d.IsDir() && strings.HasSuffix(d.Name(), ".en") &&
            strings.Contains(strings.ToLower(d.Name()), "eval") {
            if first == "" { first = path }
        }
        return nil
    })
    return first
}

// evaluateAccuracy: uses eval.en and existing vocab/emb; streams examples.
func evaluateAccuracy(gpt Transformer) (int, int) {
	seqs, err := loadEvalSequences()
	if err != nil || len(seqs) == 0 {
		return 0, 0
	}
	limit := 10000
	total, correct := 0, 0
	for _, ids := range seqs {
		if len(ids) < 2 { continue }
		// per-sequence KV caches, one per block
		type blkKV struct{ attnKV AttnKV }
		kvs := make([]blkKV, layers)
		// roll through the sequence once; predict next token at each step
		// we run up to len(ids)-1 predictions
		var yLast *mat.Dense
		for t := 0; t+1 < len(ids); t++ {
			xLast := colAsVector(emb, ids[t]) // (dModel x 1)
			yLast = xLast
			for l := 0; l < layers; l++ {
				yLast = gpt.blocks[l].ForwardLastWithKV(yLast, &kvs[l].attnKV)
			}
			logits := Unembed(yLast) // emb^T * yLast
			pred := argmaxVec(logits)
			if pred == ids[t+1] {
				correct++
			}
			total++
			if total >= limit {
				return correct, total
			}
		}
	}
	return correct, total
}

// Tokenization and files
func tokenizeENPieces(s string) []string {
    // Keep it simple: split on whitespace, then break each word into 1–4 char pieces
    // e.g., "chatgpt" -> ["chat","gpt"] ; "the" -> ["the"]
    parts := strings.Fields(strings.ToLower(s))
    out := []string{}
    for _, w := range parts {
        for len(w) > 0 {
            // prefer longest piece up to 4 bytes (ASCII assumed for now)
            take := min(4, len(w))
            // UTF-8 safety: back off to valid boundary
            for take > 1 && !utf8.ValidString(w[:take]) {
                take--
            }
            out = append(out, w[:take])
            w = w[take:]
        }
    }
    return out
}

func buildFixedVocab(tokens []string, size int) Vocab {
    if size < len(special) {
        panic("vocab size must be >= number of special tokens")
    }
    cnt := map[string]int{}
    for _, t := range tokens {
        if t == "" {
            continue
        }
        cnt[t]++
    }
    type kv struct{ k string; v int }
    arr := make([]kv, 0, len(cnt))
    for k, v := range cnt { arr = append(arr, kv{k, v}) }
    sort.Slice(arr, func(i, j int) bool {
        if arr[i].v == arr[j].v { return arr[i].k < arr[j].k }
        return arr[i].v > arr[j].v
    })
    idToToken := append([]string{}, special...)
    for _, p := range arr {
        if len(idToToken) >= size { break }
        skip := false
        for _, s := range special { if p.k == s { skip = true; break } }
        if !skip { idToToken = append(idToToken, p.k) }
    }
    for len(idToToken) < size {
        idToToken = append(idToToken, fmt.Sprintf("<pad%d>", len(idToToken)))
    }
    tok2id := map[string]int{}
    for i, t := range idToToken { tok2id[t] = i }
    return Vocab{TokenToID: tok2id, IDToToken: idToToken}
}

func findEnglishFile() string {
    candidates := []string{
       "../data/test/train.en",
        // "../data/raw/train.eng",
    }
    for _, p := range candidates {
        if fileExists(p) { return p }
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
    if len(candidates) > 0 && fileExists(candidates[0]) { return candidates[0] }
    return ""
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

func embedSequence(emb *mat.Dense, ids []int) *mat.Dense {
    d, _ := emb.Dims()
    T := len(ids)
    out := mat.NewDense(d, T, nil)
    for t, id := range ids {
        col := colAsVector(emb, id)
        for i := 0; i < d; i++ {
            out.Set(i, t, col.At(i, 0))
        }
    }
    return out
}

func Unembed(x *mat.Dense) *mat.Dense {
    if emb == nil {
        panic("embedding not initialized; call loadTrainingSet first")
    }
    return toDense(dot(emb.T(), x))
}

// -------- Tokenization (very lightweight) --------



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

// save persists the whole Transformer using the gob-based SaveTransformer.
func save(gpt Transformer) error {
	_ = os.MkdirAll("models", 0o755)
	path := "models/transformer.gob"
	return SaveTransformer(&gpt, path)
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
