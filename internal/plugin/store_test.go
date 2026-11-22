package plugin

import (
	"crypto/sha256"
	"sync"
	"testing"
	"time"

	"github.com/router-for-me/kimi-compat-plugin/internal/planner"
)

func testPlan(body []byte) planner.RequestPlan {
	return planner.RequestPlan{SourceSHA256: sha256.Sum256(body), FitsKimi: true}
}

func TestStoreGetRequiresMatchingHash(t *testing.T) {
	s := newPlanStore(8, time.Minute, time.Now)
	body := []byte(`{"a":1}`)
	s.Put("res:1", "req-1", testPlan(body))
	if _, ok := s.Get("res:1", []byte(`{"a":2}`)); ok {
		t.Fatal("stale-hash plan returned")
	}
	if _, ok := s.Get("res:1", body); !ok {
		t.Fatal("valid plan not returned")
	}
}

func TestStoreTTLExpiry(t *testing.T) {
	now := time.Now()
	clock := &now
	s := newPlanStore(8, time.Minute, func() time.Time { return *clock })
	body := []byte(`{"a":1}`)
	s.Put("res:1", "req-1", testPlan(body))
	*clock = now.Add(2 * time.Minute)
	if _, ok := s.Get("res:1", body); ok {
		t.Fatal("expired plan returned")
	}
	if s.len() != 0 {
		t.Fatal("expired entry not evicted")
	}
}

func TestStoreLRUBound(t *testing.T) {
	s := newPlanStore(2, time.Minute, time.Now)
	body := []byte(`{"a":1}`)
	s.Put("k1", "", testPlan(body))
	s.Put("k2", "", testPlan(body))
	s.Put("k3", "", testPlan(body))
	if s.len() != 2 {
		t.Fatalf("len = %d, want 2", s.len())
	}
	if _, ok := s.Get("k1", body); ok {
		t.Fatal("oldest entry not evicted")
	}
}

func TestStoreDeleteByRequestID(t *testing.T) {
	s := newPlanStore(8, time.Minute, time.Now)
	body := []byte(`{"a":1}`)
	s.Put("res:1", "req-1", testPlan(body))
	s.DeleteByRequestID("req-1")
	if _, ok := s.Get("res:1", body); ok {
		t.Fatal("plan survived request completion")
	}
	// Deleting a different request must not drop the plan.
	s.Put("res:2", "req-2", testPlan(body))
	s.DeleteByRequestID("req-outer")
	if _, ok := s.Get("res:2", body); !ok {
		t.Fatal("unrelated request deleted the plan")
	}
}

func TestStoreConcurrentAccess(t *testing.T) {
	s := newPlanStore(16, time.Minute, time.Now)
	body := []byte(`{"a":1}`)
	var wg sync.WaitGroup
	for i := 0; i < 32; i++ {
		wg.Add(1)
		go func(i int) {
			defer wg.Done()
			key := []string{"res:1", "res:2", "res:3"}[i%3]
			s.Put(key, "req", testPlan(body))
			s.Get(key, body)
			s.DeleteByRequestID("req")
			s.Delete(key)
		}(i)
	}
	wg.Wait()
}
