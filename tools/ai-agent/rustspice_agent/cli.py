"""Command-line interface for RustSpice AI Agent."""

import sys
from pathlib import Path
from typing import Optional

import click
from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.prompt import Prompt

from rustspice_agent import __version__
from rustspice_agent.client import SpiceClient, SpiceClientError
from rustspice_agent.config import Config
from rustspice_agent.formatters import format_result

console = Console()


@click.group(invoke_without_command=True)
@click.version_option(version=__version__, prog_name="rustspice-agent")
@click.option("--api-url", envvar="RUSTSPICE_API_URL", help="API server URL")
@click.pass_context
def main(ctx: click.Context, api_url: Optional[str]) -> None:
    """RustSpice AI Agent - AI-powered circuit simulation interface.

    Run without arguments to start interactive chat mode (requires AI features).
    Use subcommands for direct simulation access.
    """
    ctx.ensure_object(dict)

    # Load config
    config = Config.load()
    if api_url:
        config.api.url = api_url
    ctx.obj["config"] = config

    # If no subcommand, start interactive mode
    if ctx.invoked_subcommand is None:
        interactive_mode(config)


@main.command()
@click.argument("netlist", type=click.Path(exists=True))
@click.option("-o", "--output", type=click.Path(), help="Output file path")
@click.option("-f", "--format", "fmt", type=click.Choice(["psf", "csv", "json"]), default="csv")
@click.pass_context
def op(ctx: click.Context, netlist: str, output: Optional[str], fmt: str) -> None:
    """Run DC operating point analysis."""
    config = ctx.obj["config"]
    run_simulation(config, "op", netlist, output, fmt)


@main.command()
@click.argument("netlist", type=click.Path(exists=True))
@click.option("-s", "--source", required=True, help="Source to sweep (e.g., V1)")
@click.option("--start", type=float, required=True, help="Start value")
@click.option("--stop", type=float, required=True, help="Stop value")
@click.option("--step", type=float, required=True, help="Step size")
@click.option("-o", "--output", type=click.Path(), help="Output file path")
@click.option("-f", "--format", "fmt", type=click.Choice(["psf", "csv", "json"]), default="csv")
@click.pass_context
def dc(
    ctx: click.Context,
    netlist: str,
    source: str,
    start: float,
    stop: float,
    step: float,
    output: Optional[str],
    fmt: str,
) -> None:
    """Run DC sweep analysis."""
    config = ctx.obj["config"]
    run_simulation(
        config, "dc", netlist, output, fmt,
        source=source, start=start, stop=stop, step=step
    )


@main.command()
@click.argument("netlist", type=click.Path(exists=True))
@click.option("--tstop", type=float, required=True, help="Stop time (seconds)")
@click.option("--tstep", type=float, help="Output time step (seconds)")
@click.option("-o", "--output", type=click.Path(), help="Output file path")
@click.option("-f", "--format", "fmt", type=click.Choice(["psf", "csv", "json"]), default="csv")
@click.pass_context
def tran(
    ctx: click.Context,
    netlist: str,
    tstop: float,
    tstep: Optional[float],
    output: Optional[str],
    fmt: str,
) -> None:
    """Run transient analysis."""
    config = ctx.obj["config"]
    run_simulation(
        config, "tran", netlist, output, fmt,
        tstop=tstop, tstep=tstep
    )


@main.command()
@click.argument("netlist", type=click.Path(exists=True))
@click.option("--fstart", type=float, required=True, help="Start frequency (Hz)")
@click.option("--fstop", type=float, required=True, help="Stop frequency (Hz)")
@click.option("--points", type=int, default=10, help="Points per decade")
@click.option("-o", "--output", type=click.Path(), help="Output file path")
@click.option("-f", "--format", "fmt", type=click.Choice(["psf", "csv", "json"]), default="csv")
@click.pass_context
def ac(
    ctx: click.Context,
    netlist: str,
    fstart: float,
    fstop: float,
    points: int,
    output: Optional[str],
    fmt: str,
) -> None:
    """Run AC frequency analysis."""
    config = ctx.obj["config"]
    run_simulation(
        config, "ac", netlist, output, fmt,
        fstart=fstart, fstop=fstop, points=points
    )


@main.command()
@click.pass_context
def status(ctx: click.Context) -> None:
    """Check API server status and connection."""
    config = ctx.obj["config"]
    try:
        client = SpiceClient(base_url=config.api.url, timeout=5.0)
        if client.ping():
            console.print(f"[green]Connected to {config.api.url}[/green]")
            try:
                summary = client.get_summary()
                console.print(f"  Nodes: {summary.node_count}")
                console.print(f"  Devices: {summary.device_count}")
                console.print(f"  Models: {summary.model_count}")
            except SpiceClientError:
                console.print("  [dim]No circuit loaded[/dim]")
        else:
            console.print(f"[red]Cannot connect to {config.api.url}[/red]")
            sys.exit(1)
    except Exception as e:
        console.print(f"[red]Error: {e}[/red]")
        sys.exit(1)


@main.command()
@click.pass_context
def runs(ctx: click.Context) -> None:
    """List all simulation runs."""
    config = ctx.obj["config"]
    try:
        client = SpiceClient(base_url=config.api.url)
        runs_list = client.get_runs()
        if not runs_list:
            console.print("[dim]No simulation runs yet[/dim]")
            return
        console.print("\n[bold]Simulation Runs:[/bold]")
        for run in runs_list:
            console.print(
                f"  [{run.get('run_id')}] {run.get('analysis')} - {run.get('status')}"
            )
    except SpiceClientError as e:
        console.print(f"[red]Error: {e}[/red]")
        sys.exit(1)


def run_simulation(
    config: Config,
    analysis: str,
    netlist_path: str,
    output: Optional[str],
    fmt: str,
    **kwargs,
) -> None:
    """Execute a simulation and display/export results."""
    try:
        client = SpiceClient(base_url=config.api.url)

        # Read netlist file
        netlist_text = Path(netlist_path).read_text()

        # Run appropriate analysis
        console.print(f"[dim]Running {analysis.upper()} analysis...[/dim]")

        if analysis == "op":
            result = client.run_op(netlist=netlist_text)
        elif analysis == "dc":
            result = client.run_dc(
                source=kwargs["source"],
                start=kwargs["start"],
                stop=kwargs["stop"],
                step=kwargs["step"],
                netlist=netlist_text,
            )
        elif analysis == "tran":
            result = client.run_tran(
                tstop=kwargs["tstop"],
                tstep=kwargs.get("tstep"),
                netlist=netlist_text,
            )
        elif analysis == "ac":
            result = client.run_ac(
                fstart=kwargs["fstart"],
                fstop=kwargs["fstop"],
                points=kwargs.get("points", 10),
                netlist=netlist_text,
            )
        else:
            console.print(f"[red]Unknown analysis type: {analysis}[/red]")
            sys.exit(1)

        # Display results
        formatted = format_result(result, config.output.precision)
        console.print(Markdown(formatted))

        # Export if requested
        if output:
            client.export_run(result.run_id, output, fmt)
            console.print(f"\n[green]Results exported to {output}[/green]")

    except SpiceClientError as e:
        console.print(f"[red]Simulation error: {e}[/red]")
        sys.exit(1)
    except FileNotFoundError:
        console.print(f"[red]File not found: {netlist_path}[/red]")
        sys.exit(1)
    except Exception as e:
        console.print(f"[red]Error: {e}[/red]")
        sys.exit(1)


def interactive_mode(config: Config) -> None:
    """Start interactive chat mode with AI agent."""
    try:
        from rustspice_agent.agent import SpiceAgent, ANTHROPIC_AVAILABLE

        if not ANTHROPIC_AVAILABLE:
            console.print(
                "[yellow]AI features require the anthropic package.[/yellow]\n"
                "Install with: pip install rustspice-agent[ai]\n\n"
                "Use subcommands for direct simulation access:\n"
                "  rustspice-agent op <netlist>      - Operating point\n"
                "  rustspice-agent dc <netlist> ...  - DC sweep\n"
                "  rustspice-agent tran <netlist> .. - Transient\n"
                "  rustspice-agent ac <netlist> ...  - AC analysis\n"
            )
            sys.exit(1)

        if not config.ai.api_key:
            console.print(
                "[yellow]ANTHROPIC_API_KEY environment variable not set.[/yellow]\n"
                "Set it to use AI features, or use subcommands for direct access."
            )
            sys.exit(1)

        # Check API server connection
        client = SpiceClient(base_url=config.api.url, timeout=5.0)
        if not client.ping():
            console.print(
                f"[red]Cannot connect to RustSpice API at {config.api.url}[/red]\n"
                "Start the server with: cargo run -p sim-api"
            )
            sys.exit(1)

        agent = SpiceAgent(client=client, config=config)

        console.print(
            Panel(
                "[bold]RustSpice AI Agent[/bold]\n\n"
                "I can help you with circuit simulation. Describe your circuit or\n"
                "paste a SPICE netlist, and I'll run the appropriate analysis.\n\n"
                "Commands:\n"
                "  /help   - Show help\n"
                "  /clear  - Clear conversation\n"
                "  /quit   - Exit\n",
                title="Welcome",
                border_style="blue",
            )
        )

        while True:
            try:
                user_input = Prompt.ask("\n[bold blue]You[/bold blue]")

                if not user_input.strip():
                    continue

                # Handle commands
                if user_input.strip().lower() == "/quit":
                    console.print("[dim]Goodbye![/dim]")
                    break
                elif user_input.strip().lower() == "/clear":
                    agent.reset_conversation()
                    console.print("[dim]Conversation cleared.[/dim]")
                    continue
                elif user_input.strip().lower() == "/help":
                    console.print(
                        "\n[bold]Help:[/bold]\n"
                        "- Describe a circuit and I'll create a netlist\n"
                        "- Paste a SPICE netlist for analysis\n"
                        "- Ask about simulation results\n"
                        "- Request specific analysis types (OP, DC, TRAN, AC)\n"
                    )
                    continue

                # Process with agent
                console.print("\n[bold green]Assistant[/bold green]")
                response = agent.chat(user_input)
                console.print(Markdown(response))

            except KeyboardInterrupt:
                console.print("\n[dim]Use /quit to exit[/dim]")

    except Exception as e:
        console.print(f"[red]Error: {e}[/red]")
        sys.exit(1)


if __name__ == "__main__":
    main()
