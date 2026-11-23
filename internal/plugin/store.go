package plugin

import (
	"container/list"
	"crypto/sha256"
	"strings"
	"sync"
	"time"

	"github.com/router-for-me/kimi-fit-plugin/internal/planner"
)

// planStore keeps only edit plans — never request bodies — keyed by
// reservation ID ("res:<id>") or request ID ("req:<id>"). A reverse
// requestID -> planKey map lets request completion clean up without headers.
type planStore struct {
	mu        sync.Mutex
	entries   map[string]*list.Element
	lru       *list.List
	reqToPlan map[string]string
	max       int
	ttl       time.Duration
	now       func() time.Time
}

type planEntry struct {
	key     string
	plan    planner.RequestPlan
	expires time.Time
}

func newPlanStore(max int, ttl time.Duration, now func() time.Time) *planStore {
	if max < 1 {
		max = 128
	}
	if ttl <= 0 {
		ttl = 10 * time.Minute
	}
	if now == nil {
		now = time.Now
	}
	return &planStore{
		entries:   map[string]*list.Element{},
		lru:       list.New(),
		reqToPlan: map[string]string{},
		max:       max,
		ttl:       ttl,
		now:       now,
	}
}

func (s *planStore) Put(planKey, requestID string, plan planner.RequestPlan) {
	planKey = strings.TrimSpace(planKey)
	// The request ID may view a request buffer decoded without copying; clone
	// it so the cached reverse mapping never keeps that payload alive.
	requestID = strings.Clone(strings.TrimSpace(requestID))
	if planKey == "" {
		return
	}
	s.mu.Lock()
	defer s.mu.Unlock()
	s.evictExpiredLocked()
	if el, ok := s.entries[planKey]; ok {
		el.Value.(*planEntry).plan = plan
		el.Value.(*planEntry).expires = s.now().Add(s.ttl)
		s.lru.MoveToFront(el)
	} else {
		s.entries[planKey] = s.lru.PushFront(&planEntry{key: planKey, plan: plan, expires: s.now().Add(s.ttl)})
	}
	if requestID != "" {
		s.reqToPlan[requestID] = planKey
	}
	for len(s.entries) > s.max {
		back := s.lru.Back()
		if back == nil {
			break
		}
		s.removeLocked(back)
	}
}

// Get returns the plan only when it exists, is unexpired, and the body hash
// still matches — a stale or foreign plan must never produce edits.
func (s *planStore) Get(planKey string, body []byte) (planner.RequestPlan, bool) {
	s.mu.Lock()
	defer s.mu.Unlock()
	el, ok := s.entries[planKey]
	if !ok {
		return planner.RequestPlan{}, false
	}
	e := el.Value.(*planEntry)
	if s.now().After(e.expires) {
		s.removeLocked(el)
		return planner.RequestPlan{}, false
	}
	if sha256.Sum256(body) != e.plan.SourceSHA256 {
		return planner.RequestPlan{}, false
	}
	s.lru.MoveToFront(el)
	return e.plan, true
}

func (s *planStore) Delete(planKey string) {
	s.mu.Lock()
	defer s.mu.Unlock()
	if el, ok := s.entries[planKey]; ok {
		s.removeLocked(el)
	}
}

// DeleteByRequestID removes the plan recorded for a completing request and
// the reverse mapping. Outer-pass completions delete only outer keys; the
// nested reservation plan survives until its own completion.
func (s *planStore) DeleteByRequestID(requestID string) {
	requestID = strings.TrimSpace(requestID)
	if requestID == "" {
		return
	}
	s.mu.Lock()
	defer s.mu.Unlock()
	if planKey, ok := s.reqToPlan[requestID]; ok {
		if el, exists := s.entries[planKey]; exists {
			s.removeLocked(el)
		}
		delete(s.reqToPlan, requestID)
	}
}

func (s *planStore) removeLocked(el *list.Element) {
	e := el.Value.(*planEntry)
	delete(s.entries, e.key)
	s.lru.Remove(el)
	for reqID, planKey := range s.reqToPlan {
		if planKey == e.key {
			delete(s.reqToPlan, reqID)
		}
	}
}

func (s *planStore) evictExpiredLocked() {
	now := s.now()
	for el := s.lru.Back(); el != nil; {
		prev := el.Prev()
		if now.After(el.Value.(*planEntry).expires) {
			s.removeLocked(el)
		}
		el = prev
	}
}

func (s *planStore) len() int {
	s.mu.Lock()
	defer s.mu.Unlock()
	return len(s.entries)
}
