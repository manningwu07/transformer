package IO

import (
	"bufio"
	"errors"
	"fmt"
	"io"
	"os"
	"path/filepath"
	"strings"

	"gonum.org/v1/gonum/mat"

	"github.com/manningwu07/GPT/params"
	"github.com/manningwu07/GPT/transformer"
	"github.com/manningwu07/GPT/utils"
)

var evalCandidates = []string{
	"../data/test/wiki_eval.txt",
	// "data/test/eval.en",
	// "data/raw/eval.eng",
}

var trainCandidates = []string{
	"../data/test/wiki_train.txt",
	// "../data/raw/train.eng",
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

func findEvalFile() string {
	for _, p := range evalCandidates {
		if fileExists(p) {
			return p
		}
	}
	return fallbackFile("eval")
}

func FindTrainFile() string {
	for _, p := range trainCandidates {
		if fileExists(p) {
			return p
		}
	}
	return fallbackFile("train")
}

func fallbackFile(fileType string) string{
	// fallback: first *.en in data tree
	fmt.Println("Failed to find train candidate file, searching for first file in datatree.")
	root := "data"
	var first string
	_ = filepath.WalkDir(root, func(path string, d os.DirEntry, err error) error {
		if err == nil && !d.IsDir() && strings.HasSuffix(d.Name(), ".en") &&
			strings.Contains(strings.ToLower(d.Name()), fileType) {
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
			loss, _ := utils.CrossEntropyWithIndex(logits, ids[t+1])
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

func (it *TrainLineIter) Close() error {
	if it.f != nil {
		return it.f.Close()
	}
	return nil
}

// nextIDs returns the next line converted to token IDs, or nil, io.EOF when at end.
// When EOF is reached, the iterator rewinds to the beginning to allow multiple epochs.
func (it *TrainLineIter) NextIDs() ([]int, error) {
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
