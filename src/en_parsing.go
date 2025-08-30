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

// -------- Public functions required by your training loop --------

// loadTrainingSet builds:
//   - English 1–4 char piece vocab of size config.VocabSize
//   - tied embedding emb (dModel x |V|)
//   - prefix examples: for sentence tokens t0..tN, produce contexts [t0..ti] -> target ti1
func loadTrainingSet(gpt Transformer) ([]TrainingRecord, error) {
	if len(gpt.blocks) == 0 || gpt.blocks[0].attn == nil {
		return nil, errors.New("model is not initialized")
	}
	dModel := gpt.blocks[0].attn.dModel

	// Find English training text (re-use your existing data dir)
	p := findEnglishFile()
	if p == "" {
		return nil, errors.New("could not find English training file")
	}
	lines, err := readLines(p, 0)
	if err != nil {
		return nil, err
	}

	// Build vocab from 1–4 char pieces
	allTokens := []string{}
	seqs := make([][]string, len(lines))
	for i, s := range lines {
		toks := tokenizeENPieces(s) // 1–4 char greedy pieces
		// add BOS and EOS
		toks = append([]string{"<bos>"}, toks...)
		toks = append(toks, "<eos>")
		seqs[i] = toks
		allTokens = append(allTokens, toks...)
	}
	vocab = buildFixedVocab(allTokens, config.VocabSize)
	emb = initEmbeddings(dModel, vocab)

	// Build prefix examples
	records := make([]TrainingRecord, 0, 1<<16)
	for _, toks := range seqs {
		// convert to IDs
		ids := make([]int, len(toks))
		for i, t := range toks {
			ids[i] = vocabLookup(vocab, t)
		}
		// produce contexts up to SeqLen
		for i := 0; i+1 < len(ids); i++ {
			start := 0
			if i+1 > config.SeqLen {
				start = i + 1 - config.SeqLen
			}
			ctxIDs := ids[start : i+1] // include current token
			targetID := ids[i+1]

			X := embedSequence(emb, ctxIDs)             // (dModel x T)
			T := oneHot(len(vocab.IDToToken), targetID) // (|V| x 1)

			records = append(records, TrainingRecord{Inputs: X, Targets: T})
		}
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
		logits := Unembed(X)
		pred := argmaxVec(logits)
		tgt := argmaxVec(records[i].Targets)
		if pred == tgt {
			correct++
		}
	}
	return correct, N
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
            take := 4
            if take > len(w) {
                take = len(w)
            }
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
        // "../data/test/train.test.eval.en",
        "../data/test/train.en",
        // "../data/raw/wmt23-enzh/train.eng",
        // "../data/raw/wmt23-enzh/wmt23-enzh.en",
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
    return dot(emb.T(), x).(*mat.Dense)
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
