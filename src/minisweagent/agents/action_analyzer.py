def is_terminating(action):
    if "COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT" in action["command"]:
        return True
    return False