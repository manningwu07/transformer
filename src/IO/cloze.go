package IO

import (
    "bufio"
    "os"
    "path/filepath"
    "strings"

    "github.com/manningwu07/GPT/params"
    "github.com/manningwu07/GPT/transformer"
    "github.com/manningwu07/GPT/utils"
    "gonum.org/v1/gonum/mat"
)

var clozeCandidates = []string{
    "../data/test/cloze_eval.tsv",
    "data/test/cloze_eval.tsv",
}

func findClozeFile() string {
    for _, p := range clozeCandidates {
        if fileExists(p) {
            return p
        }
    }
    // best-effort search
    root := "data"
    var first string
    _ = filepath.WalkDir(root, func(path string, d os.DirEntry, err error) error {
        if err == nil && !d.IsDir() && strings.HasSuffix(d.Name(), ".tsv") &&
            strings.Contains(strings.ToLower(d.Name()), "cloze") {
            if first == "" {
                first = path
            }
        }
        return nil
    })
    return first
}

// EvaluateCloze computes token-level NLL on a TSV file of prompt<TAB>completion.
// Returns: nPrompts, totalTokens, sumNLL.
func EvaluateCloze(gpt transformer.Transformer, limit int) (int, int, float64) {
    path := findClozeFile()
    if path == "" {
        return 0, 0, 0
    }
    f, err := os.Open(path)
    if err != nil {
        return 0, 0, 0
    }
    defer f.Close()

    r := bufio.NewReaderSize(f, 1<<20)
    prompts := 0
    totTok := 0
    nllSum := 0.0
    for {
        line, err := r.ReadString('\n')
        if len(line) > 0 {
            if line[len(line)-1] == '\n' {
                line = line[:len(line)-1]
            }
            parts := strings.SplitN(line, "\t", 2)
            if len(parts) != 2 {
                continue
            }
            prompt := parts[0]
            completion := parts[1]

            // tokenization (lowercased ASCII by your tokenizer)
            pToks := TokenizeENPieces(prompt)
            cToks := TokenizeENPieces(completion)
            if len(cToks) == 0 {
                continue
            }
            // ids
            pIDs := make([]int, 0, len(pToks)+1)
            pIDs = append(pIDs, VocabLookup(params.Vocab, "<bos>"))
            for _, t := range pToks {
                pIDs = append(pIDs, VocabLookup(params.Vocab, t))
            }
            // KV per block
            type blkKV struct{ attnKV transformer.AttnKV }
            kvs := make([]blkKV, params.Layers)
            // prime with prompt
            var yLast *mat.Dense
            for i := 0; i < len(pIDs); i++ {
                xLast := ColAsVector(params.Emb, pIDs[i])
                yLast = AddPosCol(xLast, kvs[0].attnKV.T)
                for l := 0; l < params.Layers; l++ {
                    yLast = gpt.Blocks[l].ForwardLastWithKV(
                        yLast, &kvs[l].attnKV)
                }
            }
            // evaluate completion log-likelihood with teacher forcing
            for _, tok := range cToks {
                gold := VocabLookup(params.Vocab, tok)
                logits := Unembed(yLast)
                loss, _ := utils.CrossEntropyWithIndex(logits, gold)
                nllSum += loss
                totTok++
                // feed gold to advance
                xLast := ColAsVector(params.Emb, gold)
                yLast = AddPosCol(xLast, kvs[0].attnKV.T)
                for l := 0; l < params.Layers; l++ {
                    yLast = gpt.Blocks[l].ForwardLastWithKV(
                        yLast, &kvs[l].attnKV)
                }
            }
            prompts++
            if limit > 0 && prompts >= limit {
                break
            }
        }
        if err != nil {
            break
        }
    }
    return prompts, totTok, nllSum
}