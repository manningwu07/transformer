package IO

import (
	"bufio"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"os"
	"slices"
	"sort"

	"gonum.org/v1/gonum/mat"

	"github.com/manningwu07/GPT/params"
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
        params.PosEmb = mat.NewDense(dModel, params.Config.SeqLen, utils.RandomArray(dModel*params.Config.SeqLen, float64(dModel)))
    }

    // return line count for logging
    lines, err := countLines(p)
    if err != nil {
        return 0, err
    }
    return lines, nil
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
    // Lowercase & clean non‑ASCII
    b := make([]rune, 0, len(s))
    for _, c := range s {
        if c >= 'A' && c <= 'Z' {
            c = c + 32
        }
        if c < 0x80 {
            b = append(b, c)
        } else {
            b = append(b, ' ')
        }
    }
    text := string(b)

    // --- main change: don't strings.Fields; scan char by char
    out := make([]string, 0, len(text))
    i := 0
    for i < len(text) {
        matched := false
        // Greedy: try token lengths 4→1
        for k := 4; k >= 1; k-- {
            if i+k <= len(text) {
                piece := text[i : i+k]
                if params.Vocab.TokenToID != nil { // ensure vocab init
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
            // default: single char (covers space too)
            out = append(out, text[i:i+1])
            i++
        }
    }
    return out
}

// Stream training lines -> token IDs without loading all into memory.
type TrainLineIter struct {
	f    *os.File
	r    *bufio.Reader
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
