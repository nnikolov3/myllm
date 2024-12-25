from typing import Dict, Optional, List, Any
import textwrap
from datetime import datetime

from rich.console import Console, Group
from rich.panel import Panel
from rich.progress import (
    Progress, 
    SpinnerColumn, 
    TextColumn, 
    BarColumn, 
    TaskProgressColumn,
    TimeRemainingColumn
)
from rich.syntax import Syntax
from rich.markdown import Markdown
from rich.prompt import Prompt, Confirm
from rich.text import Text
from rich.table import Table
from rich.layout import Layout
from rich.live import Live
from rich.style import Style

class ResponseFormatter:
    """Handles formatting of responses and messages."""
    
    def __init__(self, width: int = 80):
        self.width = width
        
    def format_text(self, text: str) -> str:
        """Format text with proper wrapping and spacing."""
        paragraphs = text.split('\n\n')
        formatted_paragraphs = []
        
        for para in paragraphs:
            wrapped = textwrap.fill(para.strip(), width=self.width-5)
            formatted_paragraphs.append(wrapped)
            
        return '\n\n'.join(formatted_paragraphs)
        
    def create_response_panel(self, 
                            response: str, 
                            title: str = "Response", 
                            style: str = "green") -> Panel:
        """Create a formatted panel for responses."""
        formatted_response = self.format_text(response)
        
        styled_response = Group(
            Text(''),  # Top padding
            Markdown(formatted_response),
            Text('')   # Bottom padding
        )
        
        return Panel(
            styled_response,
            title=title,
            border_style=style,
            width=self.width,
            padding=(0, 2)
        )
        
    def create_model_responses_table(self, responses: Dict[str, str]) -> Table:
        """Create a table showing individual model responses."""
        table = Table(
            show_header=True,
            header_style="bold magenta",
            width=self.width
        )
        
        table.add_column("Model", style="cyan")
        table.add_column("Response")
        
        for model, response in responses.items():
            formatted_response = self.format_text(response)
            table.add_row(model, formatted_response)
            
        return table

class ProgressManager:
    """Manages progress bars and status indicators."""
    
    def __init__(self, console: Console):
        self.console = console
        
    def create_progress(self) -> Progress:
        """Create a progress bar with enhanced styling."""
        return Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TaskProgressColumn(),
            TimeRemainingColumn(),
            console=self.console,
            expand=True
        )
        
    def create_indeterminate_progress(self) -> Progress:
        """Create an indeterminate progress indicator."""
        return Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=self.console
        )

class StatusDisplay:
    """Manages status display and updates."""
    
    def __init__(self, console: Console):
        self.console = console
        self.layout = Layout()
        self._setup_layout()
        
    def _setup_layout(self):
        """Setup the layout structure."""
        self.layout.split(
            Layout(name="header", size=3),
            Layout(name="main"),
            Layout(name="footer", size=3)
        )
        
    def update_status(self, 
                     status: str, 
                     additional_info: Optional[Dict] = None):
        """Update the status display."""
        status_text = Text(status, style="bold blue")
        
        if additional_info:
            info_table = Table.grid(padding=1)
            for key, value in additional_info.items():
                info_table.add_row(
                    Text(f"{key}:", style="cyan"),
                    Text(str(value))
                )
            
            self.layout["main"].update(
                Group(
                    status_text,
                    Text(""),  # Spacing
                    info_table
                )
            )
        else:
            self.layout["main"].update(status_text)

class UserInterface:
    """Main interface class for the PDF chat system."""
    
    def __init__(self, width: int = 80):
        self.console = Console(width=width)
        self.formatter = ResponseFormatter(width)
        self.progress = ProgressManager(self.console)
        self.status = StatusDisplay(self.console)
        
    def startup_banner(self):
        """Display startup banner."""
        self.console.print(
            Panel.fit(
                "[bold]🤖 PDF Chat System[/bold]\n" +
                "[dim]Type 'help' for commands, 'exit' to quit[/dim]",
                border_style="blue",
                width=self.console.width
            )
        )
        
    def display_response(self, 
                        response: str, 
                        model_responses: Optional[Dict[str, str]] = None,
                        weights: Optional[Dict[str, float]] = None):
        """Display the response and optionally show model details."""
        # Display main response
        self.console.print(
            self.formatter.create_response_panel(response)
        )
        
        # Show model details if available
        if model_responses and Confirm.ask(
            "Show individual model responses?",
            default=False
        ):
            self.console.print("\n[bold]Individual Model Responses:[/bold]")
            
            if weights:
                # Show weights
                weight_table = Table(show_header=True, header_style="bold blue")
                weight_table.add_column("Model")
                weight_table.add_column("Weight")
                
                for model, weight in weights.items():
                    weight_table.add_row(
                        model,
                        f"{weight:.2%}"
                    )
                
                self.console.print(weight_table)
                self.console.print("")  # Spacing
            
            # Show responses
            self.console.print(
                self.formatter.create_model_responses_table(model_responses)
            )
            
    def get_input(self, prompt: str = "Ask a question") -> str:
        """Get user input with styling."""
        return Prompt.ask(f"\n[bold blue]{prompt}[/bold blue]")
        
    def show_error(self, error_msg: str):
        """Display error message."""
        self.console.print(
            Panel(
                self.formatter.format_text(error_msg),
                title="Error",
                border_style="red",
                width=self.console.width
            )
        )
        
    def show_help(self):
        """Display help information."""
        help_text = """
        Available Commands:
        - help: Show this help message
        - exit: Quit the program
        - clear: Clear the screen
        - stats: Show system statistics
        - models: Show model information
        
        Tips:
        - Use 'chapter X' to focus on a specific chapter
        - Ask follow-up questions for clarification
        - Request sources for specific information
        """
        
        self.console.print(
            Panel(
                self.formatter.format_text(help_text),
                title="Help",
                border_style="blue",
                width=self.console.width
            )
        )
        
    def show_statistics(self, stats: Dict[str, Any]):
        """Display system statistics."""
        stats_table = Table(show_header=True, header_style="bold magenta")
        stats_table.add_column("Metric", style="cyan")
        stats_table.add_column("Value")
        
        def add_stats_recursively(data: Dict, prefix: str = ""):
            for key, value in data.items():
                if isinstance(value, dict):
                    add_stats_recursively(value, f"{prefix}{key} ")
                else:
                    stats_table.add_row(
                        f"{prefix}{key}",
                        str(value)
                    )
        
        add_stats_recursively(stats)
        
        self.console.print(
            Panel(
                stats_table,
                title="System Statistics",
                border_style="blue",
                width=self.console.width
            )
        )
        
    async def process_with_progress(self, 
                                  message: str,
                                  coroutine: Any) -> Any:
        """Run a coroutine with a progress indicator."""
        with self.progress.create_indeterminate_progress() as progress:
            task = progress.add_task(message, total=None)
            result = await coroutine
            progress.update(task, completed=True)
            return result
            
    def clear_screen(self):
        """Clear the console screen."""
        self.console.clear()
        
    def update_status(self, 
                     status: str, 
                     additional_info: Optional[Dict] = None):
        """Update the status display."""
        self.status.update_status(status, additional_info)
        
    def start_live_display(self) -> Live:
        """Start a live display for real-time updates."""
        return Live(
            self.status.layout,
            console=self.console,
            refresh_per_second=4
        )