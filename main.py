"""
FRC Scoring Analyzer — 入口點
"""

import sys
from app import ScoringAnalyzer

if __name__ == "__main__":
    video = sys.argv[1] if len(sys.argv) > 1 else None
    app = ScoringAnalyzer(video_path=video)
    app.mainloop()
