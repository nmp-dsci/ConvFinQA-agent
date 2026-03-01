from mcp.server.fastmcp import FastMCP
from dotenv import load_dotenv

load_dotenv("~/.env")

# Create an MCP server
mcp = FastMCP(
    name="calculator",
    host="0.0.0.0",  # only used for SSE transport (localhost)
    port=8050,  # only used for SSE transport (set this to any port)
    stateless_http=True,
)

# all_ops = ["add", "subtract", "multiply", "divide", "exp", "greater", "table_max",
#            "table_min", "table_sum", "table_average"]


@mcp.tool()
def add(a: float, b: float) -> float:
    """
    Do sum / add operation with 2 numbers 
    Args:
        a: float
        b: float
    Returns: total of two numbers
    """
    return a + b

@mcp.tool()
def subtract(a: float, b: float) -> float:
    """
    Do sum / add operation with 2 numbers 
    Args:
        a: float
        b: float
    Returns: single number
    """
    return a - b

@mcp.tool()
def multiply(a: float, b: float) -> float:
    """
    Multiply to number together 
    Args:
        a: float
        b: float
    Returns: product of two numbers 
    """
    return a * b

@mcp.tool()
def divide(a: float, b: float) -> float:
    """
    Divide the numbers provided
    Args:
        a: float
        b: float
    Returns: a/ b  
    """
    return a / b

@mcp.tool()
def exp(a: float, b: float) -> float:
    """
    Exponential of a to b 
    Args:
        a: float
        b: float
    Returns:  a ** b  
    """
    return a ** b

@mcp.tool()
def greater(a: float, b: float) -> float:
    """
    if A greater than B then True else False
    Args:
        a: float
        b: float
    Returns: true or false for A > B 
    """
    return 1 if a >  b else 0 


# Run the server
if __name__ == "__main__":
    transport = "stdio"
    if transport == "stdio":
        print("Running server with stdio transport")
        mcp.run(transport="stdio")
    elif transport == "sse":
        print("Running server with SSE transport")
        mcp.run(transport="sse")
    elif transport == "streamable-http":
        print("Running server with Streamable HTTP transport")
        mcp.run(transport="streamable-http")
    else:
        raise ValueError(f"Unknown transport: {transport}")
