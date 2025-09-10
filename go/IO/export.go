package IO

import (
	"bufio"
	"encoding/binary"
	"fmt"
	"io"
	"os"

	"github.com/manningwu07/GPT/params"
)

// ExportTokenIDsBinary writes token ID sequences to a binary data file plus an index:
//
//   - .bin = concatenated int32 token sequences
//   - .idx = int64 offsets (start,length) per example
//
// It will split into shards <= maxShardBytes (e.g. 10 GB).
func ExportTokenIDsBinary(inPath, outPrefix string, maxShardBytes int64) error {
	if params.Vocab.TokenToID == nil || len(params.Vocab.IDToToken) == 0 {
        return fmt.Errorf("vocab is not initialized; load or build vocab before exporting")
    }
	inF, err := os.Open(inPath)
	if err != nil {
		return err
	}
	defer inF.Close()
	reader := bufio.NewReader(inF)

	shard := 0
	var (
		dataF *os.File
		idxF  *os.File
		wData io.Writer
		wIdx  io.Writer
		cur   int64
	)

	openShard := func() error {
		if dataF != nil {
			dataF.Close()
			idxF.Close()
		}
		dataF, err = os.Create(fmt.Sprintf("%s-%03d.bin", outPrefix, shard))
		if err != nil {
			return err
		}
		idxF, err = os.Create(fmt.Sprintf("%s-%03d.idx", outPrefix, shard))
		if err != nil {
			return err
		}
		wData = bufio.NewWriter(dataF)
		wIdx = bufio.NewWriter(idxF)
		cur = 0
		return nil
	}

	if err := openShard(); err != nil {
		return err
	}

	buf4 := make([]byte, 4)
	buf8 := make([]byte, 8)
	lineNum := 0
	for {
		line, err := reader.ReadString('\n')
		if line == "" && err == io.EOF {
			break
		}
		if line == "" && err != nil {
			return err
		}

		// tokenize line → ids
		toks := TokenizeENPieces(line)
		if len(toks) == 0 {
			continue
		}
		ids := make([]int, 0, len(toks)+2)
		ids = append(ids, VocabLookup(params.Vocab, "<bos>"))
		for _, t := range toks {
			ids = append(ids, VocabLookup(params.Vocab, t))
		}
		ids = append(ids, VocabLookup(params.Vocab, "<eos>"))

		// write offset + length to idx
		start := cur
		binary.LittleEndian.PutUint64(buf8, uint64(start))
		if _, err := wIdx.Write(buf8); err != nil {
			return err
		}
		binary.LittleEndian.PutUint64(buf8, uint64(len(ids)))
		if _, err := wIdx.Write(buf8); err != nil {
			return err
		}

		// write ids to bin
		for _, id := range ids {
			binary.LittleEndian.PutUint32(buf4, uint32(id))
			if _, err := wData.Write(buf4); err != nil {
				return err
			}
		}
		cur += int64(4 * len(ids))

		lineNum++

		// rollover if shard too big
		if cur >= maxShardBytes {
			if bw, ok := wData.(*bufio.Writer); ok {
				bw.Flush()
			}
			if bw, ok := wIdx.(*bufio.Writer); ok {
				bw.Flush()
			}
			shard++
			if err := openShard(); err != nil {
				return err
			}
		}

		if err == io.EOF {
			break
		}
	}
	if bw, ok := wData.(*bufio.Writer); ok {
		bw.Flush()
	}
	if bw, ok := wIdx.(*bufio.Writer); ok {
		bw.Flush()
	}
	return nil
}