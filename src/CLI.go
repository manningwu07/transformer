package main

import (
	"bufio"
	"fmt"
	"os"
	"strings"

	"github.com/manningwu07/GPT/IO"
	"github.com/manningwu07/GPT/params"
	"github.com/manningwu07/GPT/transformer"
	"github.com/manningwu07/GPT/utils"
	"gonum.org/v1/gonum/mat"
)

// ChatCLI
func ChatCLI(gpt *transformer.Transformer) {
	reader := bufio.NewReader(os.Stdin)
	fmt.Println("ChatGPT-like CLI. Type 'exit' to quit.")
	for {
		fmt.Print("You: ")
		input, _ := reader.ReadString('\n')
		if input == "exit" {
			break
		}
		// Generate up to 50 tokens
		out := Predict(gpt, input, 50)
		fmt.Println("Bot:", renderTokens(out))
	}
}

// Predict generates text autoregressively from an English prompt.
// It tokenizes the input, embeds it, runs it through the Transformer,
// and predicts next tokens until <eos> or maxLen.
func Predict(gpt *transformer.Transformer, input string, maxLen int) []string {
	if params.Emb == nil {
		panic("embeddings not initialized; call loadTrainingSet first")
	}

	// make sure <eos> is present at inference time
	IO.EnsureEOSToken()
	// Tokenize into 1–4 char pieces
	toks := IO.TokenizeENPieces(input)
	seq := append([]string{"<bos>"}, toks...)
	ids := make([]int, len(seq))
	for i, t := range seq {
		ids[i] = IO.VocabLookup(params.Vocab, t)
	}
	// Per-block KV caches
	type blkKV struct{ attnKV transformer.AttnKV }
	kvs := make([]blkKV, params.Layers)
	// Prime caches by rolling through the prompt tokens
	var yLast *mat.Dense
	for t := 0; t < len(ids); t++ {
		xLast := IO.ColAsVector(params.Emb, ids[t])
		yLast = IO.AddPosCol(xLast, kvs[0].attnKV.T)
		for l := 0; l < params.Layers; l++ {
			yLast = gpt.Blocks[l].ForwardLastWithKV(yLast, &kvs[l].attnKV)
			// For blocks >0, do NOT add positional embedding again.
		}
	}
	// Generate up to maxLen new tokens
	for steps := 0; steps < maxLen; steps++ {
		logits := IO.Unembed(yLast)
		probs := utils.ColVectorSoftmax(logits)
		nextID := utils.SampleFromProbs(probs, 10, 0.9)
		nextTok := params.Vocab.IDToToken[nextID]
		if nextTok == "<eos>" {
			break
		}
		ids = append(ids, nextID)
		seq = append(seq, nextTok)
		// advance one step using KV cache
		xLast := IO.ColAsVector(params.Emb, nextID)
		yLast = IO.AddPosCol(xLast, kvs[0].attnKV.T)
		for l := 0; l < params.Layers; l++ {
			yLast = gpt.Blocks[l].ForwardLastWithKV(yLast, &kvs[l].attnKV)
		}
	}
	// return generated tokens after the prompt
	fmt.Println("Input tokens:", toks)
	fmt.Println("Mapped IDs:", ids)
	return seq[1+len(toks):]
}

// renderTokens concatenates tokens and treats <eos> as newline.
func renderTokens(toks []string) string {
	if len(toks) == 0 {
		return ""
	}
	var sb strings.Builder
	for _, tk := range toks {
		if tk == "<eos>" {
			sb.WriteString("\n")
			break
		}
		sb.WriteString(tk)
	}
	return sb.String()
}
