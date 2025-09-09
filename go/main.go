package main

import (
    "flag"
    "fmt"
    "os"

    "github.com/manningwu07/GPT/IO"
    "github.com/manningwu07/GPT/params"
)

var (
    exportFlag bool
    cliFlag    bool
    forceFlag  bool
)

func init() {
    flag.BoolVar(&exportFlag, "export", false, "Export vocab and ID datasets (binary shards)")
    flag.BoolVar(&cliFlag, "cli", false, "Run ChatCLI that connects to Python server")
    flag.BoolVar(&forceFlag, "force", false, "Force re-export even if cache exists")
}

func main() {
    flag.Parse()

    if exportFlag {
        fmt.Println("Building vocab & exporting datasets...")

        vocabPath := "../data/test/vocab.json"   // fixed extension
        trainPrefix := "../data/test/wiki_train_ids"
        evalPrefix := "../data/test/wiki_eval_ids"

        // ---- Export vocab ----
        if !fileExists(vocabPath) || forceFlag {
            _, err := IO.BuildVocabAndEmbFromTrainDummy(params.Config.DModel, params.Config.VocabSize)
            if err != nil {
                panic(err)
            }
            if err := IO.ExportVocabJSON(vocabPath); err != nil {
                panic(err)
            }
            fmt.Println("✅ Exported vocab.json")
        } else {
            fmt.Println("⚡ Using cached vocab.json")
        }

        // ---- Export train shards ----
        if shardMissing(trainPrefix) || forceFlag {
            maxShardSize := int64(10 * 1024 * 1024 * 1024) // 10GB per shard
            if err := IO.ExportTokenIDsBinary("../data/raw/wiki_train.txt", trainPrefix, maxShardSize); err != nil {
                panic(err)
            }
            fmt.Println("✅ Exported train ID shards")
        } else {
            fmt.Println("⚡ Using cached train shards")
        }

        // ---- Export eval shards ----
        if shardMissing(evalPrefix) || forceFlag {
            maxShardSize := int64(2 * 1024 * 1024 * 1024) // 2GB per shard
            if err := IO.ExportTokenIDsBinary("../data/raw/wiki_eval.txt", evalPrefix, maxShardSize); err != nil {
                panic(err)
            }
            fmt.Println("✅ Exported eval ID shards")
        } else {
            fmt.Println("⚡ Using cached eval shards")
        }

        fmt.Println("✨ Export complete")
        return
    }

    if cliFlag {
        fmt.Println("Starting CLI… (make sure Python server.py is running on port 8000)")
        ChatCLI()
        return
    }

    fmt.Println("No flag passed. Use --export for preprocessing, or --cli for chat.")
}

// fileExists true if path exists
func fileExists(path string) bool {
    _, err := os.Stat(path)
    return err == nil
}

// shardMissing = true if no shard files exist yet for prefix
func shardMissing(prefix string) bool {
    glob1 := prefix + "-000.bin"
    return !fileExists(glob1)
}