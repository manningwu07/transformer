package main

import (
	"bufio"
	"bytes"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"os"
	"strings"

	"github.com/manningwu07/GPT/IO"
	"github.com/manningwu07/GPT/params"
)

func ChatCLI() {
	reader := bufio.NewReader(os.Stdin)
	fmt.Println("ChatCLI connected to Python server. Type 'exit' to quit.")
	for {
		fmt.Print("You: ")
		input, _ := reader.ReadString('\n')
		input = strings.TrimSpace(input)
		if input == "exit" {
			break
		}

		// Tokenize user input
		toks := IO.TokenizeENPieces(input)
		// Pull TokenToID and IDToToken from vocab.json

		if params.Vocab.TokenToID == nil || len(params.Vocab.IDToToken) == 0 || forceFlag {
			vocabPath := "../data/test/vocab.json"
			if err := IO.ImportVocabJSON(vocabPath); err != nil {
				panic(fmt.Sprintf("Failed to load vocab.json: %v", err))
			}
		}

		ids := []int{params.Vocab.TokenToID["<bos>"]}
		for _, t := range toks {
			ids = append(ids, IO.VocabLookup(params.Vocab, t))
		}

		req := map[string]any{
			"ids":                ids,
			"max_tokens":         40,
			"top_k":              15,
			"top_p":              0.9,
			"temperature":        0.7,
			"repetition_penalty": 1.3,
		}
		body, _ := json.Marshal(req)

		// Send to server
		resp, err := http.Post("http://127.0.0.1:8000/generate", "application/json", bytes.NewBuffer(body))
		if err != nil {
			fmt.Println("Error:", err)
			continue
		}
		defer resp.Body.Close()

		bodyBytes, _ := io.ReadAll(resp.Body)
		fmt.Println("Raw response:", string(bodyBytes))
		fmt.Println("Num of tokens:", len(string(bodyBytes)))

		var result map[string]any
		if err := json.Unmarshal(bodyBytes, &result); err != nil {
			fmt.Println("Decode error:", err)
			return
		}
		fmt.Println("Bot:", result["text"])
	}
}
