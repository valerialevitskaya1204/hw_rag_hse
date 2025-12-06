from collections import deque

class RecursiveCharacterTextSplitter:
    def __init__(self, chunk_size=100, chunk_overlap=50):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.levels = ["\n\n", "\n", " ", ""]

    def split(self, text):
        self.chunks = []
        self._split(text, deque(self.levels))
        merged = self._merge_chunks(self.chunks)

        return merged

    def _split(self, text, levels):
        if len(text) <= self.chunk_size:
            self.chunks.append(text.strip())
            return

        if not levels:
            self._split_with_overlap(text)
            return

        sep = levels[0]
        rest = deque(levels)
        rest.popleft()

        if sep == "":
            self._split_with_overlap(text)
            return

        parts = text.split(sep)

        for part in parts:
            part = part.strip()
            if not part:
                continue

            if len(part) <= self.chunk_size:
                self.chunks.append(part)
            else:
                self._split(part, rest)

    def _split_with_overlap(self, text):
        step = self.chunk_size - self.chunk_overlap
        for i in range(0, len(text), step):
            chunk = text[i:i+self.chunk_size].strip()
            if chunk:
                self.chunks.append(chunk)

    def _merge_chunks(self, chunks):
        merged = []
        current = ""

        for chunk in chunks:
            if len(current) + len(chunk) + 1 <= self.chunk_size:
                current = (current + " " + chunk).strip()
            else:
                if current:
                    merged.append(current)
                current = chunk

        if current:
            merged.append(current)

        final = []
        for block in merged:
            if not final:
                final.append(block)
            else:
                prev = final[-1]
                overlap_text = prev[-self.chunk_overlap:]
                final.append(overlap_text + " " + block)

        return final
