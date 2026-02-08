import re
import shlex

class BashParser:
    """
    Comprehensive bash command parser that handles:
    - Simple commands
    - Redirections (>, >>, <, 2>, 2>&1, etc.)
    - Heredocs (<<, <<-)
    - Pipelines (|)
    - Command concatenation (&&, ||, ;)
    - Background jobs (&)
    - Negation (!)
    """
    
    def parse(self, cmd_string):
        """
        Parse a bash command string into structured commands.
        """
        cmd_string = cmd_string.rstrip()
        
        # Extract heredocs first
        processed_cmd, heredocs = self._extract_heredocs(cmd_string)
        
        # Parse the command chain
        commands = self._parse_command_chain(processed_cmd, heredocs)
        
        return commands
    
    def _extract_heredocs(self, cmd_string):
        """
        Extract heredocs from command string.
        Returns (processed_cmd, heredocs_dict)
        """
        lines = cmd_string.split('\n')
        if len(lines) <= 1:
            return cmd_string, {}
        
        heredocs = {}
        processed_lines = []
        
        i = 0
        while i < len(lines):
            line = lines[i]
            
            # Look for heredoc operator
            heredoc_info = self._find_heredoc_in_line(line)
            
            if heredoc_info:
                delimiter, before, after = heredoc_info
                
                # Replace heredoc with placeholder
                placeholder = f"__HEREDOC_{delimiter}__"
                if after:
                    new_line = f"{before} {placeholder} {after}"
                else:
                    new_line = f"{before} {placeholder}"
                
                if new_line.strip():
                    processed_lines.append(new_line.strip())
                elif not processed_lines:
                    processed_lines.append("")
                
                # Collect heredoc content
                heredoc_content = []
                i += 1
                
                while i < len(lines):
                    current_line = lines[i]
                    stripped = current_line.rstrip('\r\n')
                    
                    # Check if this line ends the heredoc
                    if stripped == delimiter or stripped.lstrip() == delimiter:
                        break
                    
                    heredoc_content.append(stripped)
                    i += 1
                
                # Save heredoc content
                heredocs[delimiter] = '\n'.join(heredoc_content)
                
                # Skip delimiter line
                i += 1
            else:
                processed_lines.append(line)
                i += 1
        
        processed_cmd = '\n'.join(processed_lines)
        return processed_cmd, heredocs
    
    def _find_heredoc_in_line(self, line):
        """
        Find heredoc operator in a line.
        Returns (delimiter, text_before, text_after) or None.
        """
        pattern = r'(.*?)<<-?\s*([\'"]?)([^\s\'"]+)\2(.*)'
        match = re.match(pattern, line)
        
        if match:
            before = match.group(1).strip()
            delimiter = match.group(3)
            after = match.group(4).strip()
            return delimiter, before, after
        
        return None
    
    def _parse_command_chain(self, cmd_string, heredocs):
        """
        Parse a chain of commands connected by &&, ||, ;, &.
        """
        # First, restore heredoc placeholders
        cmd_string = self._restore_heredoc_placeholders(cmd_string)
        
        # Split by command operators
        parts = self._split_by_command_operators(cmd_string)
        
        commands = []
        
        for i, (cmd_part, operator) in enumerate(parts):
            # Parse this command (may contain pipeline)
            pipeline_commands = self._parse_pipeline_command(cmd_part, heredocs)
            
            for j, cmd_info in enumerate(pipeline_commands):
                cmd_info["operator"] = operator if j == 0 else None
                cmd_info["pipeline_stage"] = j + 1
                cmd_info["total_pipeline_stages"] = len(pipeline_commands)
                commands.append(cmd_info)
        
        return commands
    
    def _restore_heredoc_placeholders(self, cmd_string):
        """
        Replace heredoc placeholders with proper syntax.
        """
        def replace_placeholder(match):
            delimiter = match.group(1)
            return f"<< {delimiter}"
        
        pattern = r'__HEREDOC_([^_]+)__'
        return re.sub(pattern, replace_placeholder, cmd_string)
    
    def _split_by_command_operators(self, cmd_string):
        """
        Split command string by &&, ||, ;, & operators.
        """
        parts = []
        current = []
        depth = 0
        in_quote = None
        escaped = False
        
        i = 0
        n = len(cmd_string)
        
        while i < n:
            char = cmd_string[i]
            
            if escaped:
                current.append(char)
                escaped = False
                i += 1
                continue
            
            if char == '\\':
                escaped = True
                current.append(char)
                i += 1
                continue
            
            if in_quote:
                current.append(char)
                if char == in_quote:
                    in_quote = None
                i += 1
                continue
            
            if char in ['\'', '"']:
                in_quote = char
                current.append(char)
                i += 1
            elif char == '(':
                depth += 1
                current.append(char)
                i += 1
            elif char == ')':
                depth -= 1
                current.append(char)
                i += 1
            elif depth == 0:
                # Check for two-char operators
                if i + 1 < n and cmd_string[i:i+2] == '&&':
                    block = ''.join(current).strip()
                    if block or not parts:
                        operator = None if not parts else parts[-1][1]
                        parts.append((block, operator))
                    parts.append(('', '&&'))
                    current = []
                    i += 2
                elif i + 1 < n and cmd_string[i:i+2] == '||':
                    block = ''.join(current).strip()
                    if block or not parts:
                        operator = None if not parts else parts[-1][1]
                        parts.append((block, operator))
                    parts.append(('', '||'))
                    current = []
                    i += 2
                elif char == ';':
                    block = ''.join(current).strip()
                    if block or not parts:
                        operator = None if not parts else parts[-1][1]
                        parts.append((block, operator))
                    parts.append(('', ';'))
                    current = []
                    i += 1
                elif char == '&' and (i == 0 or cmd_string[i-1].isspace()):
                    # Check if next char is & (we already checked for &&)
                    if i + 1 < n and cmd_string[i+1] != '&':
                        block = ''.join(current).strip()
                        if block or not parts:
                            operator = None if not parts else parts[-1][1]
                            parts.append((block, operator))
                        parts.append(('', '&'))
                        current = []
                        i += 1
                    else:
                        current.append(char)
                        i += 1
                else:
                    current.append(char)
                    i += 1
            else:
                current.append(char)
                i += 1
        
        # Add last part
        if current:
            block = ''.join(current).strip()
            if block:
                operator = None if not parts else parts[-1][1]
                parts.append((block, operator))
        
        # Reconstruct
        result = []
        for i, (block, _) in enumerate(parts):
            if block:
                # Find operator for this block
                if i > 0 and not parts[i-1][0]:
                    operator = parts[i-1][1]
                else:
                    operator = None
                result.append((block, operator))
        
        return result
    
    def _parse_pipeline_command(self, cmd_string, heredocs):
        """
        Parse a command that may contain pipelines.
        """
        # Split by pipes
        pipe_parts = self._split_by_pipes(cmd_string)
        
        commands = []
        
        for pipe_part in pipe_parts:
            cmd_info = self._parse_single_command(pipe_part, heredocs)
            commands.append(cmd_info)
        
        return commands
    
    def _split_by_pipes(self, cmd_string):
        """
        Split command by pipes while respecting quotes.
        """
        parts = []
        current = []
        depth = 0
        in_quote = None
        escaped = False
        
        for char in cmd_string:
            if escaped:
                current.append(char)
                escaped = False
                continue
            
            if char == '\\':
                escaped = True
                current.append(char)
                continue
            
            if in_quote:
                current.append(char)
                if char == in_quote:
                    in_quote = None
                continue
            
            if char in ['\'', '"']:
                in_quote = char
                current.append(char)
            elif char == '(':
                depth += 1
                current.append(char)
            elif char == ')':
                depth -= 1
                current.append(char)
            elif char == '|' and depth == 0:
                parts.append(''.join(current).strip())
                current = []
            else:
                current.append(char)
        
        if current:
            parts.append(''.join(current).strip())
        
        return parts
    
    def _parse_single_command(self, cmd_string, heredocs):
        """
        Parse a single command (no pipes, no command operators).
        """
        cmd_string = cmd_string.strip()
        
        # Check for background
        background = False
        if cmd_string.endswith('&'):
            background = True
            cmd_string = cmd_string.rstrip('&').strip()
        
        # Check for negation
        negated = False
        if cmd_string.startswith('!'):
            negated = True
            cmd_string = cmd_string.lstrip('!').strip()
        
        cmd_info = {
            "command": None,
            "args": [],
            "redirections": [],
            "background": background,
            "negated": negated,
            "operator": None,
            "pipeline_stage": 1,
            "total_pipeline_stages": 1
        }
        
        # Extract redirections
        redirections, remaining = self._extract_redirections(cmd_string)
        cmd_info["redirections"] = redirections
        
        # Add heredoc content to heredoc redirections
        for redir in redirections:
            if redir.get("heredoc", False):
                delimiter = redir.get("target", "")
                if delimiter in heredocs:
                    redir["heredoc_content"] = heredocs[delimiter]
        
        # Parse the remaining command
        words = self._split_into_words(remaining)
        if words:
            cmd_info["command"] = words[0]
            cmd_info["args"] = words[1:]
        
        return cmd_info
    
    def _extract_redirections(self, cmd_string):
        """
        Extract all redirections from command string.
        """
        redirections = []
        tokens = []
        
        i = 0
        n = len(cmd_string)
        
        while i < n:
            # Skip whitespace
            while i < n and cmd_string[i].isspace():
                i += 1
            
            if i >= n:
                break
            
            # Check for redirection operators
            redir_match = re.match(r'(\d*>>?|&\d*>>?|\d*<[<&-]?|>&|<&|>>?|<<?-?)', cmd_string[i:])
            
            if redir_match:
                op = redir_match.group(0)
                i += len(op)
                
                # Skip whitespace
                while i < n and cmd_string[i].isspace():
                    i += 1
                
                # Extract target
                target = []
                in_quote = None
                escaped = False
                
                while i < n and (in_quote or not cmd_string[i].isspace()):
                    if escaped:
                        target.append(cmd_string[i])
                        escaped = False
                    elif cmd_string[i] == '\\':
                        escaped = True
                        target.append(cmd_string[i])
                    elif in_quote:
                        target.append(cmd_string[i])
                        if cmd_string[i] == in_quote:
                            in_quote = None
                    elif cmd_string[i] in ['\'', '"']:
                        in_quote = cmd_string[i]
                        target.append(cmd_string[i])
                    else:
                        target.append(cmd_string[i])
                    i += 1
                
                target_str = ''.join(target).strip()
                
                # Remove quotes
                if len(target_str) >= 2 and target_str[0] == target_str[-1] and target_str[0] in ['\'', '"']:
                    target_str = target_str[1:-1]
                
                # Get file descriptor if present
                fd = None
                if op[0].isdigit():
                    fd = int(op[0])
                elif op.startswith('&'):
                    if op[1:].isdigit():
                        fd = int(op[1:])
                
                redirections.append({
                    "op": op,
                    "fd": fd,
                    "target": target_str,
                    "heredoc": op in ['<<', '<<-']
                })
            else:
                # Regular token
                token = []
                in_quote = None
                escaped = False
                
                while i < n and (in_quote or not cmd_string[i].isspace()):
                    if escaped:
                        token.append(cmd_string[i])
                        escaped = False
                    elif cmd_string[i] == '\\':
                        escaped = True
                        token.append(cmd_string[i])
                    elif in_quote:
                        token.append(cmd_string[i])
                        if cmd_string[i] == in_quote:
                            in_quote = None
                    elif cmd_string[i] in ['\'', '"']:
                        in_quote = cmd_string[i]
                        token.append(cmd_string[i])
                    else:
                        token.append(cmd_string[i])
                    i += 1
                
                tokens.append(''.join(token))
        
        remaining = ' '.join(tokens)
        return redirections, remaining
    
    def _split_into_words(self, cmd_string):
        """
        Split command string into words.
        """
        if not cmd_string.strip():
            return []
        
        try:
            return shlex.split(cmd_string, posix=True)
        except:
            # Manual parsing
            words = []
            i = 0
            n = len(cmd_string)
            
            while i < n:
                while i < n and cmd_string[i].isspace():
                    i += 1
                
                if i >= n:
                    break
                
                word = []
                in_quote = None
                escaped = False
                
                while i < n and (in_quote or not cmd_string[i].isspace()):
                    if escaped:
                        word.append(cmd_string[i])
                        escaped = False
                    elif cmd_string[i] == '\\':
                        escaped = True
                        word.append(cmd_string[i])
                    elif in_quote:
                        word.append(cmd_string[i])
                        if cmd_string[i] == in_quote:
                            in_quote = None
                    elif cmd_string[i] in ['\'', '"']:
                        in_quote = cmd_string[i]
                        word.append(cmd_string[i])
                    else:
                        word.append(cmd_string[i])
                    i += 1
                
                word_str = ''.join(word)
                if len(word_str) >= 2 and word_str[0] == word_str[-1] and word_str[0] in ['\'', '"']:
                    word_str = word_str[1:-1]
                
                if word_str:
                    words.append(word_str)
            
            return words