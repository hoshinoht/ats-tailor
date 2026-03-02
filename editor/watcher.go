package main

import (
	"encoding/json"
	"log"
	"path/filepath"
	"strings"
	"sync"
	"time"

	"github.com/fsnotify/fsnotify"
)

type sseEvent struct {
	File  string `json:"file"`
	Mtime int64  `json:"mtime"`
}

type Watcher struct {
	fsw        *fsnotify.Watcher
	mu         sync.Mutex
	suppress   map[string]time.Time
	subs       map[chan []byte]struct{}
}

func newWatcher(dir string) (*Watcher, error) {
	fsw, err := fsnotify.NewWatcher()
	if err != nil {
		return nil, err
	}
	if err := fsw.Add(dir); err != nil {
		fsw.Close()
		return nil, err
	}
	return &Watcher{
		fsw:      fsw,
		suppress: make(map[string]time.Time),
		subs:     make(map[chan []byte]struct{}),
	}, nil
}

func (w *Watcher) Close() {
	w.fsw.Close()
}

// SuppressNext marks a file name (without extension) to suppress the next change event.
// Used to avoid echo events when the editor itself writes the file.
func (w *Watcher) SuppressNext(name string) {
	w.mu.Lock()
	defer w.mu.Unlock()
	w.suppress[name] = time.Now().Add(2 * time.Second)
}

func (w *Watcher) isSuppressed(name string) bool {
	w.mu.Lock()
	defer w.mu.Unlock()
	deadline, ok := w.suppress[name]
	if !ok {
		return false
	}
	if time.Now().Before(deadline) {
		delete(w.suppress, name)
		return true
	}
	delete(w.suppress, name)
	return false
}

func (w *Watcher) Subscribe() chan []byte {
	ch := make(chan []byte, 8)
	w.mu.Lock()
	w.subs[ch] = struct{}{}
	w.mu.Unlock()
	return ch
}

func (w *Watcher) Unsubscribe(ch chan []byte) {
	w.mu.Lock()
	delete(w.subs, ch)
	w.mu.Unlock()
	close(ch)
}

func (w *Watcher) broadcast(data []byte) {
	w.mu.Lock()
	defer w.mu.Unlock()
	for ch := range w.subs {
		select {
		case ch <- data:
		default:
			// slow subscriber — drop
		}
	}
}

func (w *Watcher) Run() {
	for {
		select {
		case event, ok := <-w.fsw.Events:
			if !ok {
				return
			}
			if event.Op&(fsnotify.Write|fsnotify.Create) == 0 {
				continue
			}
			base := filepath.Base(event.Name)
			if !strings.HasSuffix(base, ".yaml") {
				continue
			}
			name := strings.TrimSuffix(base, ".yaml")
			if !isAllowed(name) {
				continue
			}
			if w.isSuppressed(name) {
				continue
			}
			evt := sseEvent{File: name, Mtime: time.Now().UnixMilli()}
			data, err := json.Marshal(evt)
			if err != nil {
				log.Printf("watcher marshal error: %v", err)
				continue
			}
			w.broadcast(data)

		case err, ok := <-w.fsw.Errors:
			if !ok {
				return
			}
			log.Printf("watcher error: %v", err)
		}
	}
}
