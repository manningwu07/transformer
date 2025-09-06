package IO

import (
	"slices"
	"bufio"
	"errors"
	"fmt"
	"io"
	"os"
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
	return mat.NewDense(r, 1, dst)

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

// Helper function for buildFixedVocabFromFile
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

    // --- NEW: ensure all printable ASCII single chars are in vocab ---
    for c := 32; c <= 126; c++ {
        tok := string(rune(c))
        idToToken = append(idToToken, tok)
    }

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
        }
        if p.k == "" {
            continue
        }
        // don’t duplicate if already added as single ASCII
        already := slices.Contains(idToToken, p.k)
        if !already {
            idToToken = append(idToToken, p.k)
        }
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

// ASCII helper
func isASCIIString(s string) bool {
	for i := 0; i < len(s); i++ {
		if s[i] >= 0x80 {
			return false
		}
	}
	return true
}

// Tokenization (ASCII-only pieces). Lowercases, drops any non 1-byte ASCII chars.
func TokenizeENPieces(s string) []string {
	// Fast ASCII lowercase + drop non-ASCII (replace by space)
    b := make([]byte, 0, len(s))
    for i := 0; i < len(s); i++ {
        c := s[i]
        if c >= 'A' && c <= 'Z' {
            c = c + 32
        }
        if c < 0x80 {
            b = append(b, c)
        } else {
            b = append(b, ' ')
        }
    }
    parts := strings.Fields(string(b))

    // If vocab is not initialized yet (during vocab building), fall back to
    // fixed 1–4 slicing to produce counts.
    if params.Vocab.TokenToID == nil {
        out := make([]string, 0, 16)
        for _, w := range parts {
            for len(w) > 0 {
                take := min(4, len(w))
                out = append(out, w[:take])
                w = w[take:]
            }
        }
        return out
    }

    // Vocab-aware greedy encoding: for each word, take the longest piece in
    // vocab (4→3→2→1). Guarantees coverage by falling back to single chars.
    out := make([]string, 0, 16)
    for _, w := range parts {
        i := 0
        for i < len(w) {
            take := 1
            // try 4→3→2→1
            for k := 4; k >= 1; k-- {
                if i+k <= len(w) {
                    cand := w[i : i+k]
                    if _, ok := params.Vocab.TokenToID[cand]; ok {
                        take = k
                        break
                    }
                }
            }
            out = append(out, w[i:i+take])
            i += take
        }
    }
    return out
}

// Stream training lines -> token IDs without loading all into memory.
type TrainLineIter struct {
	path string
	f    *os.File
	r    *bufio.Reader
}

func NewTrainLineIter(path string) (*TrainLineIter, error) {
	f, err := os.Open(path)
	if err != nil {
		return nil, err
	}
	return &TrainLineIter{path: path, f: f, r: bufio.NewReaderSize(f, 1<<20)}, nil
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
