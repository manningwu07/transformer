package IO

import (
	"bufio"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"math"
	"math/rand/v2"
	"os"
	"slices"
	"sort"

	"gonum.org/v1/gonum/mat"

	"github.com/manningwu07/GPT/params"
)

// TrainingRecord is a generic dataset example (inputs column vector,
// targets column vector, both mat.Dense wrappers are easy to create).
type TrainingRecord struct {
	Inputs  *mat.Dense
	Targets *mat.Dense
}

// Special tokens kept at the start of the vocab
var special = []string{"<pad>", "<bos>", "<eos>", "<unk>"}

func ExportVocabJSON(path string) error {
	f, err := os.Create(path)
	if err != nil {
		return err
	}
	defer f.Close()
	data := map[string]any{
		"TokenToID": params.Vocab.TokenToID,
		"IDToToken": params.Vocab.IDToToken,
	}
	enc := json.NewEncoder(f)
	enc.SetIndent("", "  ")
	return enc.Encode(data)
}

// BuildVocabAndEmbFromTrainDummy just builds vocab and embeddings without needing Go Transformer
func BuildVocabAndEmbFromTrainDummy(dModel, vocabSize int) (int, error) {
    p := FindTrainFile()
    if p == "" {
        return 0, errors.New("could not find training file")
    }
    var err error
    params.Vocab, _, err = buildFixedVocabFromFile(p, vocabSize)
    if err != nil {
        return 0, err
    }
    params.Emb = initEmbeddings(dModel, params.Vocab)

    if params.PosEmb == nil || params.PosEmb.RawMatrix().Rows != dModel || params.PosEmb.RawMatrix().Cols != params.Config.SeqLen {
        params.PosEmb = mat.NewDense(dModel, params.Config.SeqLen, randomArray(dModel*params.Config.SeqLen, float64(dModel)))
    }

    // return line count for logging
    lines, err := countLines(p)
    if err != nil {
        return 0, err
    }
    return lines, nil
}

func VocabLookup(v params.Vocabulary, tok string) int {
	 if v.TokenToID == nil {
        panic("VocabLookup called with nil vocab. Load/build vocab first.")
    }
    if id, ok := v.TokenToID[tok]; ok {
        return id
    }
    return v.TokenToID["<unk>"]
}

// Initialize embeddings with small random values.
// Shape: (dModel x |V|)
func initEmbeddings(dModel int, v params.Vocabulary) *mat.Dense {
	data := randomArray(dModel*len(v.IDToToken), float64(dModel))
	return mat.NewDense(dModel, len(v.IDToToken), data)
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
			toks := TokenizeForVocab(line)
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

    // Buckets: length → list of (token,count)
    buckets := map[int][]kv{
        1: {}, 2: {}, 3: {}, 4: {},
    }

    // --- cutoff ---
    const minCount = 10 // 🛑 frequency cutoff: skip n-grams < 5 occurrences

    for k, v := range cnt {
        L := len(k)
        if L >= 1 && L <= 4 && v >= minCount {
            buckets[L] = append(buckets[L], kv{k, v})
        }
    }

    // Sort each bucket by frequency desc
    for L := 1; L <= 4; L++ {
        sort.Slice(buckets[L], func(i, j int) bool {
            if buckets[L][i].v == buckets[L][j].v {
                return buckets[L][i].k < buckets[L][j].k
            }
            return buckets[L][i].v > buckets[L][j].v
        })
    }

    // Start vocab with specials
    idToToken := append([]string{}, special...)

    // Always guarantee ASCII single chars are in vocab
    for c := 32; c <= 126; c++ {
        tok := string(rune(c))
        idToToken = append(idToToken, tok)
    }

    // Compute remaining budget
    remaining := size - len(idToToken)
    if remaining <= 0 {
        return finalizeVocab(idToToken, size)
    }

    // Divide evenly across lengths 2,3,4
    share := remaining / 3
    target2 := len(idToToken) + share
    target3 := target2 + share
    target4 := target3 + share

    addBucket := func(L int, limit int) {
        for _, p := range buckets[L] {
            if len(idToToken) >= limit || len(idToToken) >= size {
                return
            }
            if !isASCIIString(p.k) || p.k == "" {
                continue
            }
            if slices.Contains(idToToken, p.k) {
                continue
            }
            idToToken = append(idToToken, p.k)
        }
    }

    addBucket(2, target2)
    addBucket(3, target3)
    addBucket(4, target4)

    return finalizeVocab(idToToken, size)
}

func finalizeVocab(idToToken []string, size int) params.Vocabulary {
    for len(idToToken) < size {
        idToToken = append(idToToken, fmt.Sprintf("<pad%d>", len(idToToken)))
    }
    tok2id := make(map[string]int, len(idToToken))
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

// TokenizeForVocab is used during vocab building.
// It lowercases, removes non-ASCII, and greedily emits substrings of length 1..4.
func TokenizeForVocab(s string) []string {
    // Lowercase & clean to ASCII
    b := make([]rune, 0, len(s))
    for _, c := range s {
        if c >= 'A' && c <= 'Z' {
            c += 32
        }
        if c < 0x80 {
            b = append(b, c)
        } else {
            b = append(b, ' ')
        }
    }
    text := string(b)

    out := make([]string, 0, len(text))
    i := 0
    for i < len(text) {
        matched := false
        // Greedy: try token lengths 4→1
        for k := 4; k >= 1; k-- {
            if i+k <= len(text) {
                piece := text[i : i+k]
                // just take the substring as a vocab candidate
                out = append(out, piece)
                i += k
                matched = true
                break
            }
        }
        if !matched {
            // shouldn't ever happen, but safety
            i++
        }
    }
    return out
}

// TokenizeENPieces is used at runtime after vocab has been built.
// It attempts greedy matching against existing vocab; if not found, falls back.
func TokenizeENPieces(s string) []string {
    // Lowercase & clean to ASCII
    b := make([]rune, 0, len(s))
    for _, c := range s {
        if c >= 'A' && c <= 'Z' {
            c += 32
        }
        if c < 0x80 {
            b = append(b, c)
        } else {
            b = append(b, ' ')
        }
    }
    text := string(b)

    out := make([]string, 0, len(text))
    i := 0
    for i < len(text) {
        matched := false
        // Greedy: try token lengths 4→1
        for k := 4; k >= 1; k-- {
            if i+k <= len(text) {
                piece := text[i : i+k]
                if params.Vocab.TokenToID != nil {
                    if _, ok := params.Vocab.TokenToID[piece]; ok {
                        out = append(out, piece)
                        i += k
                        matched = true
                        break
                    }
                }
            }
        }
        if !matched {
            // fallback: single char (or could map to <unk>)
            out = append(out, text[i:i+1])
            i++
        }
    }
    return out
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


// Helper function
func randomArray(size int, v float64) []float64 {
	min := -1.0 / math.Sqrt(v+1e-12)
	max := 1.0 / math.Sqrt(v+1e-12)
	out := make([]float64, size)
	for i := 0; i < size; i++ {
		out[i] = min + (max-min)*rand.Float64()
	}
	return out
}