package IO

import (
    "encoding/json"
    "fmt"
    "os"
    "path/filepath"

    tk "github.com/sugarme/tokenizer"
    "github.com/sugarme/tokenizer/models"
    "github.com/sugarme/tokenizer/normalizers"
    "github.com/sugarme/tokenizer/pretokenizers"
    "github.com/sugarme/tokenizer/processors"
    "github.com/sugarme/tokenizer/trainers"

    "github.com/manningwu07/GPT/params"
)

// Global tokenizer used by exporter/CLI
var bpeTokenizer *tk.Tokenizer

// TrainOrLoadBPE trains a BPE tokenizer on corpusPath (if tokenizer.json not found)
// and loads it into memory. It also fills params.Vocab with TokenToID/IDToToken.
func TrainOrLoadBPE(corpusPath, tokPath string, vocabSize int) error {
    if fileExists(tokPath) {
        t, err := tk.FromFile(tokPath)
        if err != nil {
            return err
        }
        bpeTokenizer = t
        return fillParamsVocabFromTokenizer()
    }

    // Model + components (ByteLevel BPE-like; uses whitespace split if ByteLevel unavailable)
    bpe := models.NewBPE()
    t := tk.NewTokenizer(bpe)

    // Normalize to NFKC lower for English
    t.WithNormalizer(normalizers.NewSequence(
        normalizers.NewNFKC(),
        normalizers.NewLowercase(),
    ))
    // Pretokenizer: whitespace is robust and simple; you can switch to ByteLevel later
    t.WithPreTokenizer(pretokenizers.NewWhitespaceSplit())

    // Add BOS/EOS/PAD/UNK handling via template processor
    // We will still add BOS/EOS in exporter for consistency, but this keeps decode robust.
    proc := processors.NewTemplateProcessing(
        "<bos> $A <eos>",
        "$A",
        map[string]int{
            "<bos>": 1,
            "<eos>": 2,
        },
    )
    t.WithPostProcessor(proc)

    tr := trainers.NewBpeTrainer()
    tr.VocabSize = vocabSize
    tr.SpecialTokens = []string{"<pad>", "<bos>", "<eos>", "<unk>"}

    if err := t.Train(tr, []string{corpusPath}); err != nil {
        return err
    }
    if err := os.MkdirAll(filepath.Dir(tokPath), 0o755); err != nil {
        return err
    }
    if err := t.Save(tokPath); err != nil {
        return err
    }
    bpeTokenizer = t
    return fillParamsVocabFromTokenizer()
}

func fillParamsVocabFromTokenizer() error {
    if bpeTokenizer == nil {
        return fmt.Errorf("tokenizer not initialized")
    }
    vocab := bpeTokenizer.GetVocab(true)
    // Build IDToToken in index order 0..N-1
    id2tok := make([]string, len(vocab))
    tok2id := make(map[string]int, len(vocab))
    for tok, id := range vocab {
        tok2id[tok] = id
        id2tok[id] = tok
    }
    params.Vocab = params.Vocabulary{TokenToID: tok2id, IDToToken: id2tok}
    return nil
}

// EncodeBPE encodes raw text into token IDs (without BOS/EOS).
func EncodeBPE(text string) ([]int, error) {
    if bpeTokenizer == nil {
        return nil, fmt.Errorf("tokenizer not initialized")
    }
    enc, err := bpeTokenizer.EncodeSingle(text)
    if err != nil {
        return nil, err
    }
    ids := enc.Ids
    out := make([]int, len(ids))
    for i, v := range ids {
        out[i] = int(v)
    }
    return out, nil
}

// ExportVocabJSONBPE writes TokenToID/IDToToken from the loaded tokenizer.
func ExportVocabJSONBPE(path string) error {
    if bpeTokenizer == nil {
        return fmt.Errorf("tokenizer not initialized")
    }
    data := map[string]any{
        "TokenToID": params.Vocab.TokenToID,
        "IDToToken": params.Vocab.IDToToken,
    }
    f, err := os.Create(path)
    if err != nil {
        return err
    }
    defer f.Close()
    enc := json.NewEncoder(f)
    enc.SetIndent("", "  ")
    return enc.Encode(data)
}