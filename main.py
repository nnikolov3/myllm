#!/usr/bin/env python3
import asyncio
import argparse
import os
import sys
from typing import Optional
import signal
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.console import Console
import traceback
from rich import print as rprint

# Add the project root directory to Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.models.chat import MultiModelChat
from src.models.processor import PDFProcessor
from src.utils.resources import ResourceManager
from src.utils.logging import LogManager, get_logger
from src.ui.interface import UserInterface

console = Console()

class PDFChatApplication:
    """Main application class for PDF Chat system."""
    
    def __init__(self, debug_mode: bool = False):
        console.print("[bold blue]Initializing PDF Chat System...[/bold blue]")
        
        # Initialize logging
        self.log_manager = LogManager()
        self.log_manager.setup_logging(debug_mode=debug_mode)
        self.logger = get_logger(__name__)
        
        # Initialize components
        try:
            console.print("[yellow]Initializing resource manager...[/yellow]")
            self.resource_manager = ResourceManager()
            console.print("[green]Resource manager initialized[/green]")
        except Exception as e:
            console.print(f"[red]Failed to initialize resource manager: {str(e)}[/red]")
            raise
            
        self.ui = UserInterface()
        self.processor: Optional[PDFProcessor] = None
        self.chat: Optional[MultiModelChat] = None
        
        # Set up signal handlers
        signal.signal(signal.SIGINT, self.handle_interrupt)
        signal.signal(signal.SIGTERM, self.handle_interrupt)

    def handle_interrupt(self, signum, frame):
        """Handle interrupt signals gracefully."""
        self.logger.info("Received interrupt signal, cleaning up...")
        if hasattr(self, 'resource_manager'):
            self.resource_manager.stop_monitoring()
        sys.exit(0)
        
    # In PDFChatApplication class, update the initialize method:

    async def initialize(self, pdf_path: str):
        """Initialize the application with a PDF file."""
        try:
            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                console=console
            ) as progress:
                # Start resource monitoring
                task = progress.add_task("Starting resource monitoring...", total=1)
                try:
                    await self.resource_manager.start_monitoring()
                    progress.update(task, completed=1)
                except Exception as e:
                    console.print(f"[red]Error starting resource monitoring: {str(e)}[/red]")
                    return False
                
                # Initialize processor
                task = progress.add_task("Initializing document processor...", total=1)
                try:
                    self.processor = PDFProcessor(self.resource_manager)
                    progress.update(task, completed=1)
                except Exception as e:
                    console.print(f"[red]Error initializing processor: {str(e)}[/red]")
                    return False
                
                # Initialize chat system
                task = progress.add_task("Initializing chat system...", total=1)
                try:
                    self.chat = MultiModelChat(self.resource_manager)
                    progress.update(task, completed=1)
                except Exception as e:
                    console.print(f"[red]Error initializing chat system: {str(e)}[/red]")
                    return False
                
                # Load PDF
                task = progress.add_task(f"Loading PDF: {pdf_path}", total=1)
                try:
                    docs_generator = self.processor.load_pdf_lazy(pdf_path)
                    progress.update(task, completed=1)
                except Exception as e:
                    console.print(f"[red]Error loading PDF: {str(e)}[/red]")
                    return False
                
                # Process document
                task = progress.add_task("Processing document content...", total=1)
                try:
                    console.print("[yellow]Processing document, this may take a few minutes...[/yellow]")
                    processed_docs = await self.processor.process_documents(docs_generator)
                    if not processed_docs:
                        raise ValueError("No documents were processed")
                    progress.update(task, completed=1)
                except Exception as e:
                    console.print(f"[red]Error processing documents: {str(e)}[/red]")
                    console.print(traceback.format_exc())
                    return False
                
                # Initialize chat with documents
                task = progress.add_task("Setting up chat system...", total=1)
                try:
                    await self.chat.initialize(processed_docs)
                    progress.update(task, completed=1)
                except Exception as e:
                    console.print(f"[red]Error initializing chat with documents: {str(e)}[/red]")
                    console.print(traceback.format_exc())
                    return False
                
                # Get statistics
                try:
                    stats = self.processor.get_chapter_statistics()
                    console.print(f"[green]Successfully processed {stats['total_chapters']} chapters[/green]")
                except Exception as e:
                    console.print(f"[yellow]Warning: Could not get statistics: {str(e)}[/yellow]")
                
            console.print("[bold green]System initialization complete![/bold green]")
            console.print("\n[bold blue]Ready for questions. Type 'help' for commands, 'exit' to quit.[/bold blue]")
            return True
            
        except Exception as e:
            self.logger.error(f"Initialization error: {e}")
            console.print(f"[bold red]Failed to initialize: {str(e)}[/bold red]")
            console.print(traceback.format_exc())
            return False

    async def process_query(self, query: str):
        """Process a user query and display the response."""
        try:
            console.print("[yellow]Processing query...[/yellow]")
            
            # Get response from chat system
            response = await self.chat.get_response(query)
            
            # Display response
            self.ui.display_response(
                response["combined_response"],
                response["individual_responses"],
                response["weights_used"]
            )
            
        except Exception as e:
            self.logger.error(f"Query processing error: {e}")
            console.print(f"[bold red]Error processing query: {str(e)}[/bold red]")
            
    async def run_chat_loop(self):
        """Run the main chat loop."""
        self.ui.startup_banner()
        
        while True:
            try:
                # Get user input
                query = self.ui.get_input()
                
                # Handle commands
                if query.lower() == 'exit':
                    break
                elif query.lower() == 'help':
                    self.ui.show_help()
                    continue
                elif query.lower() == 'clear':
                    self.ui.clear_screen()
                    continue
                elif query.lower() == 'stats':
                    stats = {
                        'Document': self.processor.get_chapter_statistics(),
                        'System': self.resource_manager.get_resource_summary()
                    }
                    self.ui.show_statistics(stats)
                    continue
                    
                # Process regular query
                await self.process_query(query)
                
            except Exception as e:
                self.logger.error(f"Chat loop error: {e}")
                console.print(f"[bold red]An error occurred: {str(e)}[/bold red]")

async def main():
    """Main application entry point."""
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='PDF Chat System')
    parser.add_argument('pdf_path', help='Path to the PDF file')
    parser.add_argument('-d', '--debug', action='store_true', help='Enable debug mode')
    args = parser.parse_args()
    
    # Validate PDF path
    if not os.path.exists(args.pdf_path):
        console.print(f"[bold red]Error: File not found: {args.pdf_path}[/bold red]")
        return 1
        
    # Initialize application
    app = PDFChatApplication(debug_mode=args.debug)
    
    try:
        # Initialize with PDF
        if not await app.initialize(args.pdf_path):
            return 1
            
        # Run chat loop
        await app.run_chat_loop()
        
    except Exception as e:
        app.logger.error(f"Application error: {e}")
        console.print(f"[bold red]A fatal error occurred: {str(e)}[/bold red]")
        return 1
        
    finally:
        app.resource_manager.stop_monitoring()
        console.print("[yellow]Shutting down...[/yellow]")
        
    return 0

if __name__ == "__main__":
    try:
        exit_code = asyncio.run(main())
        exit(exit_code)
    except KeyboardInterrupt:
        console.print("\n[yellow]Application terminated by user[/yellow]")
        exit(0)
    except Exception as e:
        console.print(f"[bold red]Unhandled error: {str(e)}[/bold red]")
        exit(1)