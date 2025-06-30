#!/bin/bash
# Smart cd function that auto-activates conda environments

smart_cd() {
    # Change directory first
    cd "$1"
    
    # Check if there's a .envrc file and auto-activate
    if [[ -f .envrc ]]; then
        echo "🔧 Auto-activating environment..."
        source .envrc
    fi
}

# Usage: smart_cd /path/to/worktree
# Add this to your ~/.bashrc: alias cd=smart_cd
