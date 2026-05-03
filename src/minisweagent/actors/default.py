from pydantic import BaseModel
from minisweagent import Model
import re
from minisweagent.agents.default import FormatError

class ActorConfig(BaseModel):
    system_message: str
    instance_message: str
    timeout_template: str
    format_error_template: str
    action_regex: str = r"```bash\s*\n(.*?)\n```"
    
class DefaultActor:
    def __init__(self, model: Model, config_class: type = ActorConfig, **kwargs):
        self.model = model
        self.config = config_class(**kwargs)
    
    def _step_to_message(self, step: dict) -> dict:
        return [
            {"role": "assistant", "content": step["action"]},
            {"role": "user", "content": step["observation"]},
        ]
        
    def _preprocess_messages(self, trajectory: list, summaries: list) -> list:    
        # incremental rolling compaction
        messages = [
            {"role": "system", "content": self.config.system_message},
            {"role": "user", "content": self.config.instance_message},
        ]
        chunk_size = self.config.recent_steps_limit
        total_steps = len(trajectory)
        x = total_steps // chunk_size
        x = max(0, x-1)
        recent_steps = total_steps - x*chunk_size

        # Don't summarize the first chunk until we have at least 2 full chunks to summarize= 
                
        if recent_steps != chunk_size and total_steps != recent_steps:
            # Convert to 1d Array and back to get the last recent_steps steps as messages
            recent_msgs = trajectory[-recent_steps:].map(self._step_to_message).flatten().tolist()
            if len(summaries) > 0 and total_steps - len(summaries)*chunk_size >= chunk_size:
                return messages + [
                    {   
                        "role": "user",      
                        "content": f"[CONTEXT SUMMARY - DO NOT REPEAT VERBATIM]\n" + summaries[-1],
                    }
                ] + recent_msgs
            return messages + recent_msgs

        # Case: recent_steps == chunk_size
        old_msgs    = trajectory[-2*chunk_size:-chunk_size].map(self._step_to_message).flatten().tolist()
        recent_msgs = trajectory[-chunk_size:].map(self._step_to_message).flatten().tolist()

        if total_steps - len(summaries)*chunk_size == chunk_size and len(summaries) > 0:
            pass
        else:
            new_summary = self._summarize(old_msgs, summaries[-1])
            summaries.append(new_summary)
                    
        return messages + [
            {   
                "role": "user",      
                "content": f"[CONTEXT SUMMARY - DO NOT REPEAT VERBATIM]\n" + summaries[-1],
            }
        ] + recent_msgs


    def generate_action(self, messages: list[dict]) -> list[str]:
        response = self.model.query(self.messages)
        if "echo COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT" in response["content"]:
            response["content"] = response["content"].replace("echo COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT", "echo COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT && git add -A && git diff --cached")
        
        return self.parse_action(response)
    
    def parse_action(self, response: dict) -> dict:
        """Parse the action from the message. Returns the action."""
        actions = re.findall(self.config.action_regex, response["content"], re.DOTALL)
        if len(actions) == 1:
            action = actions[0].strip()
            return {"action": action, **response}
        raise FormatError(self.render_template(self.config.format_error_template, actions=actions))