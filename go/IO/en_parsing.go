package IO

import (
	"fmt"
	"io"
	"os"
	"path/filepath"
	"strings"

	"github.com/manningwu07/GPT/params"
)

var trainCandidates = []string{
	"../data/raw/wiki_train.txt",
	// "../data/raw/train.eng",
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

func fileExists(p string) bool {
	_, err := os.Stat(p)
	return err == nil
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
