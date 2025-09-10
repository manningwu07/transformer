package IO

import (
	"encoding/json"
	"fmt"
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

// ImportVocabJSON loads vocab.json into params.Vocab
func ImportVocabJSON(path string) error {
    f, err := os.Open(path)
    if err != nil {
        return err
    }
    defer f.Close()
    var data struct {
        TokenToID map[string]int `json:"TokenToID"`
        IDToToken []string       `json:"IDToToken"`
    }
    if err := json.NewDecoder(f).Decode(&data); err != nil {
        return err
    }
    params.Vocab = params.Vocabulary{
        TokenToID: data.TokenToID,
        IDToToken: data.IDToToken,
    }
    return nil
}

