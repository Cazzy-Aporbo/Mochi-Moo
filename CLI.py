"""
Command-line interface for Mochi-Moo
Author: Cazandra Aporbo MS
"""

import click
import asyncio
import json
from pathlib import Path
from rich.console import Console
from rich.table import Table
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.syntax import Syntax
from rich.panel import Panel
from rich.layout import Layout
from rich import print as rprint

from mochi_moo.core import MochiCore, CognitiveMode

console = Console()


@click.group()
@click.version_option(version='1.0.0')
def cli():
    """
    Mochi-Moo CLI - Interact with the pastel singularity from your terminal
    """
    pass


@cli.command()
@click.argument('input_text')
@click.option('--mode', '-m', default='standard', help='Cognitive mode')
@click.option('--domains', '-d', multiple=True, help='Domains for synthesis')
@click.option('--visualize', '-v', help='Visualization type')
@click.option('--emotional/--no-emotional', default=True, help='Track emotional context')
@click.option('--output', '-o', type=click.Path(), help='Output file')
def process(input_text, mode, domains, visualize, emotional, output):
    """
    Process text through Mochi's cognitive systems
    """
    async def run():
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console
        ) as progress:
            task = progress.add_task("[cyan]Initializing Mochi...", total=None)

            mochi = MochiCore()
            mochi.set_mode(mode)

            progress.update(task, description="[green]Processing your thoughts...")

            response = await mochi.process(
                input_text,
                emotional_context=emotional,
                visualization=visualize,
                domains=list(domains) if domains else None
            )

            progress.update(task, description="[magenta]Rendering response...")

        # Create beautiful output
        panel = Panel(
            response['content'],
            title="[bold magenta]Mochi's Response[/bold magenta]",
            border_style="cyan",
            padding=(1, 2)
        )
        console.print(panel)

        if 'micro_dose' in response:
            console.print(f"\n[italic dim]Micro-dose: {response['micro_dose']}[/italic dim]")

        if output:
            with open(output, 'w') as f:
                json.dump(response, f, indent=2)
            console.print(f"\n[green]Response saved to {output}[/green]")

    asyncio.run(run())


@cli.command()
@click.argument('domains', nargs=-1, required=True)
@click.argument('query')
def synthesize(domains, query):
    """
    Perform cross-domain synthesis
    """
    mochi = MochiCore()
    result = mochi.synthesizer.integrate(list(domains), query)

    table = Table(title="Cross-Domain Synthesis Results")
    table.add_column("Metric", style="cyan")
    table.add_column("Value", style="magenta")

    table.add_row("Coherence Score", f"{result['coherence_score']:.2f}")
    table.add_row("Insights Found", str(len(result['primary_insights'])))
    table.add_row("Patterns Identified", str(len(result['cross_domain_patterns'])))

    console.print(table)

    if result['primary_insights']:
        console.print("\n[bold]Primary Insights:[/bold]")
        for i, insight in enumerate(result['primary_insights'][:3], 1):
            console.print(f"  {i}. {insight}")


@cli.command()
def interactive():
    """
    Start interactive Mochi session
    """
    async def run():
        mochi = MochiCore()

        console.print("[bold cyan]Welcome to Mochi-Moo Interactive Mode[/bold cyan]")
        console.print("[dim]Type 'exit' to quit, 'mode <name>' to switch modes[/dim]\n")

        while True:
            try:
                user_input = console.input("[bold green]You:[/bold green] ")

                if user_input.lower() == 'exit':
                    console.print("[yellow]Goodbye! Dream in pastel...[/yellow]")
                    break

                if user_input.lower().startswith('mode '):
                    mode_name = user_input[5:]
                    mochi.set_mode(mode_name)
                    console.print(f"[cyan]Switched to {mode_name} mode[/cyan]")
                    continue

                response = await mochi.process(user_input)

                console.print(f"\n[bold magenta]Mochi:[/bold magenta] {response['content']}")

                if 'micro_dose' in response:
                    console.print(f"\n[italic dim]{response['micro_dose']}[/italic dim]")

                console.print()

            except KeyboardInterrupt:
                console.print("\n[yellow]Session interrupted[/yellow]")
                break
            except Exception as e:
                console.print(f"[red]Error: {e}[/red]")

    asyncio.run(run())


@cli.command()
def server():
    """
    Start the Mochi-Moo API server
    """
    from mochi_moo.server import run_server
    console.print("[bold green]Starting Mochi-Moo API server...[/bold green]")
    run_server()


@cli.command()
def status():
    """
    Show Mochi system status
    """
    mochi = MochiCore()

    layout = Layout()
    layout.split_column(
        Layout(name="header", size=3),
        Layout(name="body"),
        Layout(name="footer", size=3)
    )

    layout["header"].update(
        Panel("[bold cyan]Mochi-Moo System Status[/bold cyan]", border_style="cyan")
    )

    status_table = Table(show_header=False, box=None)
    status_table.add_column("Property", style="cyan")
    status_table.add_column("Value", style="magenta")

    status_table.add_row("Current Mode", mochi.current_mode.value)
    status_table.add_row("Foresight Depth", str(mochi.foresight.depth))
    status_table.add_row("Synthesis Cache Size", str(len(mochi.synthesizer.synthesis_cache)))
    status_table.add_row("Interaction History", str(len(mochi.interaction_history)))

    emotional_state = mochi.get_emotional_state()
    for key, value in emotional_state.items():
        status_table.add_row(key.replace('_', ' ').title(), f"{value:.2f}")

    layout["body"].update(Panel(status_table, title="System Metrics", border_style="green"))
    layout["footer"].update(
        Panel("[dim]Ready to dream in pastel[/dim]", border_style="magenta")
    )

    console.print(layout)


def main():
    """
    Main entry point for CLI
    """
    cli()

