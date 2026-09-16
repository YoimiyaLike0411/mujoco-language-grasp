"""Run the unchanged formal evaluator with separate contact audit files."""
import sys
from pathlib import Path

import validate_seeds
from contact_diagnostic import TraceEnv


class AuditEnv(TraceEnv):
    def step(self, command):
        episode = self.task.episode_id
        if getattr(self, "_audit_episode", None) != episode:
            if hasattr(self, "trace"):
                self.trace.close()
            self._audit_episode = episode
            output = Path(sys.argv[sys.argv.index("--output") + 1])
            output.mkdir(parents=True, exist_ok=True)
            self.trace = (output / (episode + "_contacts.jsonl")).open("w")
        return super().step(command)

    def close(self):
        if hasattr(self, "trace"):
            self.trace.close()
        super().close()

if __name__ == "__main__":
    validate_seeds.GraspEnv = AuditEnv
    validate_seeds.main()
