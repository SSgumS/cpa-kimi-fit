package planner

import (
	"errors"
	"fmt"
)

// scan.go is a minimal JSON parser that records byte spans instead of
// materializing values. Editing a 2.5 MB request body needs element offsets,
// not a decoded tree, so strings keep their raw span (content between quotes)
// and are never decoded unless a caller asks.

type nodeKind int

const (
	kindObject nodeKind = iota
	kindArray
	kindString
	kindScalar
)

type member struct {
	key   string
	value *jnode
}

// jnode is one JSON value. start..end spans the whole value (for a string,
// including the surrounding quotes); strStart..strEnd is the content span
// inside the quotes for kindString.
type jnode struct {
	kind              nodeKind
	start, end        int
	strStart, strEnd  int
	members           []member
	elems             []*jnode
}

func (n *jnode) field(key string) *jnode {
	if n == nil || n.kind != kindObject {
		return nil
	}
	for _, m := range n.members {
		if m.key == key {
			return m.value
		}
	}
	return nil
}

func (n *jnode) fieldIs(body []byte, key, want string) bool {
	v := n.field(key)
	if v == nil || v.kind != kindString {
		return false
	}
	// Values may carry escapes; compare the decoded content.
	s, err := decodeString(body[v.start:v.end])
	return err == nil && s == want
}

var errJSONDepth = errors.New("json depth limit exceeded")

const maxJSONDepth = 512

// parseJSON scans body and returns the root node with full span metadata.
// It is strict: malformed input returns an error and planning must fail open.
func parseJSON(body []byte) (*jnode, error) {
	p := &scanner{body: body}
	p.skipWS()
	n, err := p.value(0)
	if err != nil {
		return nil, err
	}
	p.skipWS()
	if p.pos != len(body) {
		return nil, fmt.Errorf("trailing data at offset %d", p.pos)
	}
	return n, nil
}

type scanner struct {
	body []byte
	pos  int
}

func (s *scanner) skipWS() {
	for s.pos < len(s.body) {
		switch s.body[s.pos] {
		case ' ', '\t', '\n', '\r':
			s.pos++
		default:
			return
		}
	}
}

func (s *scanner) value(depth int) (*jnode, error) {
	if depth > maxJSONDepth {
		return nil, errJSONDepth
	}
	if s.pos >= len(s.body) {
		return nil, fmt.Errorf("unexpected end of input at %d", s.pos)
	}
	switch c := s.body[s.pos]; {
	case c == '{':
		return s.object(depth)
	case c == '[':
		return s.array(depth)
	case c == '"':
		return s.str()
	case c == 't' || c == 'f' || c == 'n':
		return s.literal()
	case c == '-' || (c >= '0' && c <= '9'):
		return s.number()
	default:
		return nil, fmt.Errorf("unexpected byte %q at %d", c, s.pos)
	}
}

func (s *scanner) object(depth int) (*jnode, error) {
	n := &jnode{kind: kindObject, start: s.pos}
	s.pos++ // '{'
	s.skipWS()
	if s.pos < len(s.body) && s.body[s.pos] == '}' {
		s.pos++
		n.end = s.pos
		return n, nil
	}
	for {
		s.skipWS()
		if s.pos >= len(s.body) || s.body[s.pos] != '"' {
			return nil, fmt.Errorf("object key expected at %d", s.pos)
		}
		keyNode, err := s.str()
		if err != nil {
			return nil, err
		}
		key, err := decodeString(s.body[keyNode.start:keyNode.end])
		if err != nil {
			return nil, err
		}
		s.skipWS()
		if s.pos >= len(s.body) || s.body[s.pos] != ':' {
			return nil, fmt.Errorf("':' expected at %d", s.pos)
		}
		s.pos++
		s.skipWS()
		val, err := s.value(depth + 1)
		if err != nil {
			return nil, err
		}
		n.members = append(n.members, member{key: key, value: val})
		s.skipWS()
		if s.pos >= len(s.body) {
			return nil, fmt.Errorf("unterminated object at %d", s.pos)
		}
		switch s.body[s.pos] {
		case ',':
			s.pos++
			continue
		case '}':
			s.pos++
			n.end = s.pos
			return n, nil
		default:
			return nil, fmt.Errorf("unexpected byte %q in object at %d", s.body[s.pos], s.pos)
		}
	}
}

func (s *scanner) array(depth int) (*jnode, error) {
	n := &jnode{kind: kindArray, start: s.pos}
	s.pos++ // '['
	s.skipWS()
	if s.pos < len(s.body) && s.body[s.pos] == ']' {
		s.pos++
		n.end = s.pos
		return n, nil
	}
	for {
		s.skipWS()
		val, err := s.value(depth + 1)
		if err != nil {
			return nil, err
		}
		n.elems = append(n.elems, val)
		s.skipWS()
		if s.pos >= len(s.body) {
			return nil, fmt.Errorf("unterminated array at %d", s.pos)
		}
		switch s.body[s.pos] {
		case ',':
			s.pos++
			continue
		case ']':
			s.pos++
			n.end = s.pos
			return n, nil
		default:
			return nil, fmt.Errorf("unexpected byte %q in array at %d", s.body[s.pos], s.pos)
		}
	}
}

func (s *scanner) str() (*jnode, error) {
	n := &jnode{kind: kindString, start: s.pos}
	s.pos++ // '"'
	n.strStart = s.pos
	for s.pos < len(s.body) {
		switch c := s.body[s.pos]; {
		case c == '"':
			n.strEnd = s.pos
			s.pos++
			n.end = s.pos
			return n, nil
		case c == '\\':
			s.pos++
			if s.pos >= len(s.body) {
				return nil, fmt.Errorf("dangling escape at %d", s.pos-1)
			}
			switch e := s.body[s.pos]; e {
			case '"', '\\', '/', 'b', 'f', 'n', 'r', 't':
				s.pos++
			case 'u':
				if s.pos+5 > len(s.body) {
					return nil, fmt.Errorf("short \\u escape at %d", s.pos)
				}
				if _, err := decodeU(s.body[s.pos+1 : s.pos+5]); err != nil {
					return nil, err
				}
				s.pos += 5
			default:
				return nil, fmt.Errorf("invalid escape \\%c at %d", e, s.pos)
			}
		case c < 0x20:
			return nil, fmt.Errorf("unescaped control byte %#x at %d", c, s.pos)
		default:
			s.pos++
		}
	}
	return nil, fmt.Errorf("unterminated string at %d", n.start)
}

func (s *scanner) literal() (*jnode, error) {
	n := &jnode{kind: kindScalar, start: s.pos}
	for _, lit := range []string{"true", "false", "null"} {
		if s.pos+len(lit) <= len(s.body) && string(s.body[s.pos:s.pos+len(lit)]) == lit {
			s.pos += len(lit)
			n.end = s.pos
			return n, nil
		}
	}
	return nil, fmt.Errorf("invalid literal at %d", s.pos)
}

func (s *scanner) number() (*jnode, error) {
	n := &jnode{kind: kindScalar, start: s.pos}
	if s.body[s.pos] == '-' {
		s.pos++
	}
	if s.pos >= len(s.body) || !digit(s.body[s.pos]) {
		return nil, fmt.Errorf("invalid number at %d", n.start)
	}
	if s.body[s.pos] == '0' {
		s.pos++ // leading zero: no further digits allowed by JSON grammar
	} else {
		for s.pos < len(s.body) && digit(s.body[s.pos]) {
			s.pos++
		}
	}
	if s.pos < len(s.body) && s.body[s.pos] == '.' {
		s.pos++
		if s.pos >= len(s.body) || !digit(s.body[s.pos]) {
			return nil, fmt.Errorf("invalid fraction at %d", s.pos)
		}
		for s.pos < len(s.body) && digit(s.body[s.pos]) {
			s.pos++
		}
	}
	if s.pos < len(s.body) && (s.body[s.pos] == 'e' || s.body[s.pos] == 'E') {
		s.pos++
		if s.pos < len(s.body) && (s.body[s.pos] == '+' || s.body[s.pos] == '-') {
			s.pos++
		}
		if s.pos >= len(s.body) || !digit(s.body[s.pos]) {
			return nil, fmt.Errorf("invalid exponent at %d", s.pos)
		}
		for s.pos < len(s.body) && digit(s.body[s.pos]) {
			s.pos++
		}
	}
	n.end = s.pos
	return n, nil
}

func digit(c byte) bool { return c >= '0' && c <= '9' }

// decodeString unescapes one JSON string literal (raw span including quotes).
// Correct for any size; callers with hot paths should prefer raw spans for
// large values (image payloads) and decode only when the content is needed.
func decodeString(raw []byte) (string, error) {
	if len(raw) < 2 || raw[0] != '"' || raw[len(raw)-1] != '"' {
		return "", fmt.Errorf("not a quoted string")
	}
	inner := raw[1 : len(raw)-1]
	// Fast path: no escapes.
	hasEsc := false
	for _, b := range inner {
		if b == '\\' {
			hasEsc = true
			break
		}
	}
	if !hasEsc {
		return string(inner), nil
	}
	out := make([]byte, 0, len(inner))
	for i := 0; i < len(inner); i++ {
		c := inner[i]
		if c != '\\' {
			out = append(out, c)
			continue
		}
		i++
		if i >= len(inner) {
			return "", fmt.Errorf("dangling escape")
		}
		switch inner[i] {
		case '"', '\\', '/':
			out = append(out, inner[i])
		case 'b':
			out = append(out, '\b')
		case 'f':
			out = append(out, '\f')
		case 'n':
			out = append(out, '\n')
		case 'r':
			out = append(out, '\r')
		case 't':
			out = append(out, '\t')
		case 'u':
			if i+4 >= len(inner) {
				return "", fmt.Errorf("short \\u escape")
			}
			cp, err := decodeU(inner[i+1 : i+5])
			if err != nil {
				return "", err
			}
			i += 4
			// Surrogate pair.
			if cp >= 0xD800 && cp <= 0xDBFF && i+6 <= len(inner)-1 && inner[i+1] == '\\' && inner[i+2] == 'u' {
				lo, err := decodeU(inner[i+3 : i+7])
				if err == nil && lo >= 0xDC00 && lo <= 0xDFFF {
					i += 6
					cp = 0x10000 + ((cp - 0xD800) << 10) + (lo - 0xDC00)
				}
			}
			out = append(out, []byte(string(rune(cp)))...)
		default:
			return "", fmt.Errorf("bad escape \\%c", inner[i])
		}
	}
	return string(out), nil
}

func decodeU(hex []byte) (int, error) {
	v := 0
	for _, c := range hex {
		v <<= 4
		switch {
		case c >= '0' && c <= '9':
			v += int(c - '0')
		case c >= 'a' && c <= 'f':
			v += int(c-'a') + 10
		case c >= 'A' && c <= 'F':
			v += int(c-'A') + 10
		default:
			return 0, fmt.Errorf("bad hex digit %q", c)
		}
	}
	return v, nil
}
