"""
Page 2: Technical Description
Displays technical documentation from project_summary.txt
"""

import streamlit as st
import os
import re


def show():
    st.title("🔬 Technical Description")
    st.markdown("---")
    
    # Read project summary
    summary_path = "project_summary.txt"
    
    if os.path.exists(summary_path):
        with open(summary_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Display the content with formatting
        st.markdown("### Project Summary")
        st.markdown("This page displays the complete technical documentation of the project.")
        st.markdown("---")
        
        # Process content to convert section headers properly
        lines = content.split('\n')
        processed_lines = []
        i = 0
        
        while i < len(lines):
            line = lines[i].strip()
            
            # Check if this is a separator line (line with ===)
            if line.startswith('=') and len(line) > 10:
                # Look ahead to find the title (next non-empty, non-=== line)
                j = i + 1
                while j < len(lines) and (not lines[j].strip() or lines[j].strip().startswith('=')):
                    j += 1
                
                if j < len(lines):
                    title_line = lines[j].strip()
                    # Check if next line is also === (title is between two === lines)
                    if j + 1 < len(lines) and lines[j + 1].strip().startswith('='):
                        processed_lines.append(f"\n### {title_line}\n")
                        i = j + 2  # Skip title and second === line
                        continue
                
                # If no title found between === lines, just skip this line
                i += 1
                continue
            
            # Regular content - preserve original line
            processed_lines.append(lines[i])
            i += 1
        
        # Join all lines and display as a single markdown block
        processed_content = '\n'.join(processed_lines)
        st.markdown(processed_content)
        
    else:
        st.error(f"Project summary file not found at: {summary_path}")
        st.info("Please ensure that `project_summary.txt` exists in the project root directory.")
    
    st.markdown("---")
    st.markdown("**Note**: This content is automatically loaded from `project_summary.txt`")
