from fastapi import APIRouter, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
from utils import add_log, LOG_QUEUE, logger
from config import (
    SimulationSettings, PromptSettings,
    GRID_SIZE, NUM_ENTITIES, MAX_STEPS, CHEBYSHEV_DISTANCE,
    LLM_MODEL, LLM_MAX_TOKENS, LLM_TEMPERATURE,
    REQUEST_DELAY, MAX_CONCURRENT_REQUESTS,
    DEFAULT_MESSAGE_GENERATION_PROMPT, DEFAULT_MEMORY_GENERATION_PROMPT, DEFAULT_MOVEMENT_GENERATION_PROMPT
)

# Define the models for settings and prompts
class CustomRuleRequest(BaseModel):
    rule: str

# A list to store custom rules and installed plugins
custom_rules = []
installed_plugins = []

# Create a new APIRouter for customization-related endpoints
router = APIRouter()

### Custom Rules
@router.get("/custom_rules")
async def custom_rules_docs():
    """
    Comprehensive inline documentation for custom rules creation and usage.
    """
    html_content = """
    <html>
        <head>
            <title>Custom Rules Documentation</title>
            <style>
                body {
                    font-family: Arial, sans-serif;
                    line-height: 1.6;
                    margin: 20px;
                    background-color: #f9f9f9;
                    color: #333;
                }
                h1 {
                    color: #2c3e50;
                }
                h2 {
                    margin-top: 20px;
                    color: #34495e;
                }
                pre {
                    background: #f4f4f4;
                    padding: 10px;
                    border: 1px solid #ddd;
                    border-radius: 4px;
                    overflow-x: auto;
                }
                .example, .note {
                    padding: 15px;
                    border-radius: 4px;
                }
                .example {
                    background-color: #eaf7ff;
                    border-left: 4px solid #3498db;
                }
                .note {
                    background-color: #fff8e1;
                    border-left: 4px solid #f39c12;
                }
                a {
                    color: #2980b9;
                    text-decoration: none;
                }
                a:hover {
                    text-decoration: underline;
                }
            </style>
        </head>
        <body>
            <h1>Custom Rules Documentation</h1>
            <p>Define specific behaviors for your simulation entities using custom rules. These rules allow dynamic adjustments to entity and environment behavior.</p>

            <h2>Getting Started</h2>
            <p>Write rules in Python and use the following objects for logic:</p>
            <ul>
                <li><strong>entity:</strong> Represents the current agent in the simulation.</li>
                <li><strong>simulation:</strong> Represents the overall simulation environment.</li>
                <li><strong>log(message):</strong> Log messages to the simulation's audit trail.</li>
            </ul>

            <h2>Examples</h2>
            <div class="example">
                <h3>1. Boundary Check</h3>
                <p>Add a memory entry when an entity reaches the boundary:</p>
                <pre>
if entity.x == 0 or entity.x == simulation.width - 1:
    entity.memory.append("Boundary reached")
                </pre>
            </div>
            <div class="example">
                <h3>2. Collision Detection</h3>
                <p>Log interactions when two entities collide:</p>
                <pre>
if entity1.x == entity2.x and entity1.y == entity2.y:
    log(f"Entity {entity1.id} collided with Entity {entity2.id}")
                </pre>
            </div>

            <h2>System Variables</h2>
            <ul>
                <li><strong>entity:</strong> Access the entity's attributes such as <code>entity.x</code>, <code>entity.y</code>, and <code>entity.memory</code>.</li>
                <li><strong>simulation:</strong> Use properties like <code>simulation.width</code> and <code>simulation.height</code> to reference the environment.</li>
            </ul>

            <h2>Validation</h2>
            <div class="note">
                <strong>Important:</strong> All rules are validated for syntax and restricted to safe operations. Avoid using:
                <ul>
                    <li>File system operations (e.g., open, write).</li>
                    <li>External network calls.</li>
                    <li>Execution of arbitrary system commands.</li>
                </ul>
            </div>

            <h2>How to Use</h2>
            <ol>
                <li>Go to the <a href="/docs">API Documentation</a>.</li>
                <li>Use the <code>/simulation/custom_rules</code> endpoint to add rules.</li>
                <li>Submit a JSON payload like this:
                <pre>
{
    "rule": "if entity.x == 10: entity.memory.append('At boundary')"
}
                </pre>
                </li>
            </ol>

            <h2>Management</h2>
            <ul>
                <li><strong>Add a Rule:</strong> Use <code>POST /custom_rules</code>.</li>
                <li><strong>List All Rules:</strong> Use <code>GET /custom_rules</code>.</li>
                <li><strong>Delete a Rule:</strong> Use <code>DELETE /custom_rules/{rule_id}</code>.</li>
            </ul>

            <h2>Advanced Use Cases</h2>
            <ul>
                <li>Trigger events based on conditions like proximity or thresholds.</li>
                <li>Implement multi-agent collaboration through shared states or memory.</li>
                <li>Use rules to modify environment behavior dynamically.</li>
            </ul>
        </body>
    </html>
    """
    return HTMLResponse(content=html_content)

@router.post("/custom_rules", tags=["Customization"])
async def add_custom_rule(custom_rule_request: CustomRuleRequest):
    """
    Add or update custom rules for the simulation.

    <br>
    <h2>Custom Rules Documentation</h2>
    <p>
        Custom rules allow you to extend the simulation behavior dynamically.
        Use Python syntax and predefined objects such as <code>entity</code> and <code>environment</code>.
    </p>
    <h3>Quick Instructions</h3>
    <ul>
        <li>Write rules in valid Python syntax.</li>
        <li>Use the <code>entity</code> object to access or modify individual agent properties.</li>
        <li>Use the <code>environment</code> object to interact with the simulation's environment.</li>
    </ul>
    <h3>Example Rule</h3>
    <pre>
    if entity.x == 10:
        entity.memory += 'At boundary'
    </pre>
    <p>
        For more details, visit the full 
        <a href="/customization/custom_rules" target="_blank">Custom Rules Documentation</a>.
    </p>
    """
    try:
        rule = custom_rule_request.rule  # Access the rule from the parsed body
        # Validate rule (basic syntax check)
        try:
            compile(rule, "<string>", "exec")
        except SyntaxError as e:
            raise HTTPException(status_code=400, detail=f"Invalid Python syntax: {str(e)}")
        
        custom_rules.append(rule)
        return {"status": "Rule added successfully", "rule": rule, "total_rules": len(custom_rules)}

    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error adding custom rule: {str(e)}")

### Plugin Management
@router.get("/manage_plugins")
async def plugins_docs():
    """

    Comprehensive inline documentation for plugin management.
    """
    html_content = """
    <html>
        <head>
            <title>Plugin Management Documentation</title>
            <style>
                body {
                    font-family: Arial, sans-serif;
                    line-height: 1.6;
                    margin: 20px;
                    background-color: #f4f4f9;
                    color: #333;
                }
                h1 {
                    color: #0056b3;
                }
                h2 {
                    margin-top: 20px;
                    color: #0066cc;
                }
                pre {
                    background: #eee;
                    padding: 10px;
                    border: 1px solid #ccc;
                    overflow-x: auto;
                }
                .example {
                    background-color: #e8f6fc;
                    padding: 15px;
                    border-left: 4px solid #00aaff;
                }
                .note {
                    background-color: #fff4e5;
                    padding: 15px;
                    border-left: 4px solid #ffa726;
                }
            </style>
        </head>
        <body>
            <h1>Plugin Management Documentation</h1>
            <p>Manage plugins or extensions for your simulation environment. Plugins allow you to extend the functionality of your simulation by adding custom features or behaviors.</p>

            <h2>Supported Actions</h2>
            <p>The following actions are supported for plugin management:</p>
            <ul>
                <li><strong>Install:</strong> Adds a plugin to the simulation.</li>
                <li><strong>Uninstall:</strong> Removes a plugin from the simulation.</li>
                <li><strong>List:</strong> Lists all currently installed plugins.</li>
            </ul>

            <h2>Example Payloads</h2>
            <div class="example">
                <h3>1. Installing a Plugin</h3>
                <pre>
{
    "plugin_name": "custom_logger",
    "action": "install"
}
                </pre>
                <p>This example installs a plugin named <code>custom_logger</code>.</p>
            </div>
            <div class="example">
                <h3>2. Uninstalling a Plugin</h3>
                <pre>
{
    "plugin_name": "custom_logger",
    "action": "uninstall"
}
                </pre>
                <p>This example removes the plugin named <code>custom_logger</code>.</p>
            </div>
            <div class="example">
                <h3>3. Listing Installed Plugins</h3>
                <pre>
{
    "action": "list"
}
                </pre>
                <p>This example retrieves a list of all currently installed plugins.</p>
            </div>

            <h2>Usage Instructions</h2>
            <ol>
                <li>Go to the <a href="/docs">API Documentation</a>.</li>
                <li>Use the <code>/plugins</code> endpoint.</li>
                <li>Submit a JSON payload with the desired action and plugin name (if applicable).</li>
            </ol>

            <h2>Validation Rules</h2>
            <p>The following validation rules are applied for plugin management:</p>
            <ul>
                <li><strong>Action:</strong> Must be one of <code>install</code>, <code>uninstall</code>, or <code>list</code>.</li>
                <li><strong>Plugin Name:</strong> Required for <code>install</code> and <code>uninstall</code> actions.</li>
                <li><strong>Unique Plugins:</strong> Plugins must have unique names. Duplicate installations are not allowed.</li>
            </ul>

            <div class="note">
                <strong>Note:</strong> Plugin functionality depends on the specific implementation of each plugin. Ensure that any custom plugins are compatible with the simulation environment.
            </div>

            <h2>Advanced Use Cases</h2>
            <ul>
                <li>Create a logging plugin to track simulation events in real-time.</li>
                <li>Develop an AI plugin to enable agents to make intelligent decisions.</li>
                <li>Integrate visualization plugins for enhanced simulation analysis.</li>
            </ul>
        </body>
    </html>
    """
    return HTMLResponse(content=html_content)


@router.post("/manage_plugins", tags=["Customization"])
async def manage_plugins(plugin_name: str = None, action: str = None):
    """
    Manage plugins or extensions for the simulation environment.

    <br>
    <h2>Plugin Management Documentation</h2>
    <p>
        Plugins extend the simulation's capabilities. This endpoint allows you to install, uninstall, or list plugins.
    </p>
    <h3>Quick Instructions</h3>
    <ul>
        <li>Use the "install" action to add a plugin.</li>
        <li>Use the "uninstall" action to remove a plugin.</li>
        <li>Use the "list" action to see all installed plugins.</li>
    </ul>
    <h3>Example Payloads</h3>
    <ul>
        <li><b>Install:</b> <code>{"plugin_name": "custom_logger", "action": "install"}</code></li>
        <li><b>Uninstall:</b> <code>{"plugin_name": "custom_logger", "action": "uninstall"}</code></li>
        <li><b>List:</b> <code>{"action": "list"}</code></li>
    </ul>
    <p>
        For more details, visit the full 
        <a href="/customization/manage_plugins" target="_blank">Plugin Management Documentation</a>.
    </p>
    """
    try:
        if action not in ["install", "uninstall", "list"]:
            raise HTTPException(status_code=400, detail="Invalid action. Supported actions: install, uninstall, list.")

        if action == "list":
            return {"installed_plugins": installed_plugins}

        if action in ["install", "uninstall"] and not plugin_name:
            raise HTTPException(status_code=400, detail="Plugin name is required for install or uninstall actions.")

        if action == "install":
            if plugin_name in installed_plugins:
                raise HTTPException(status_code=400, detail=f"Plugin '{plugin_name}' is already installed.")
            installed_plugins.append(plugin_name)
            return {"status": "Plugin installed successfully", "plugin_name": plugin_name}

        if action == "uninstall":
            if plugin_name not in installed_plugins:
                raise HTTPException(status_code=400, detail=f"Plugin '{plugin_name}' is not installed.")
            installed_plugins.remove(plugin_name)
            return {"status": "Plugin uninstalled successfully", "plugin_name": plugin_name}

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error managing plugins: {str(e)}")

### Simulation Settings
@router.get("/settings", response_model=SimulationSettings, tags=["Customization"])
async def get_settings():
    try:
        add_log("Fetching simulation settings.")
        return SimulationSettings(
            grid_size=GRID_SIZE,
            num_entities=NUM_ENTITIES,
            max_steps=MAX_STEPS,
            chebyshev_distance=CHEBYSHEV_DISTANCE,
            llm_model=LLM_MODEL,
            llm_max_tokens=LLM_MAX_TOKENS,
            llm_temperature=LLM_TEMPERATURE,
            request_delay=REQUEST_DELAY,
            max_concurrent_requests=MAX_CONCURRENT_REQUESTS
        )
    except Exception as e:
        add_log(f"Error fetching simulation settings: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/settings", response_model=SimulationSettings, tags=["Customization"])
async def set_settings(settings: SimulationSettings):
    try:
        add_log("Updating simulation settings.")
        global GRID_SIZE, NUM_ENTITIES, MAX_STEPS, CHEBYSHEV_DISTANCE
        global LLM_MODEL, LLM_MAX_TOKENS, LLM_TEMPERATURE, REQUEST_DELAY, MAX_CONCURRENT_REQUESTS

        GRID_SIZE = settings.grid_size
        NUM_ENTITIES = settings.num_entities
        MAX_STEPS = settings.max_steps
        CHEBYSHEV_DISTANCE = settings.chebyshev_distance
        LLM_MODEL = settings.llm_model
        LLM_MAX_TOKENS = settings.llm_max_tokens
        LLM_TEMPERATURE = settings.llm_temperature
        REQUEST_DELAY = settings.request_delay
        MAX_CONCURRENT_REQUESTS = settings.max_concurrent_requests
        return settings
    except Exception as e:
        add_log(f"Error updating simulation settings: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


### Prompt Templates
@router.get("/prompts", response_model=PromptSettings, tags=["Customization"])
async def get_prompts():
    try:
        add_log("Fetching current prompt templates.")
        return PromptSettings(
            message_generation_prompt=DEFAULT_MESSAGE_GENERATION_PROMPT,
            memory_generation_prompt=DEFAULT_MEMORY_GENERATION_PROMPT,
            movement_generation_prompt=DEFAULT_MOVEMENT_GENERATION_PROMPT
        )
    except Exception as e:
        add_log(f"Error fetching prompt templates: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/prompts", response_model=PromptSettings, tags=["Customization"])
async def set_prompts(prompts: PromptSettings):
    try:
        add_log("Updating prompt templates.")
        global DEFAULT_MESSAGE_GENERATION_PROMPT, DEFAULT_MEMORY_GENERATION_PROMPT, DEFAULT_MOVEMENT_GENERATION_PROMPT

        DEFAULT_MESSAGE_GENERATION_PROMPT = prompts.message_generation_prompt
        DEFAULT_MEMORY_GENERATION_PROMPT = prompts.memory_generation_prompt
        DEFAULT_MOVEMENT_GENERATION_PROMPT = prompts.movement_generation_prompt
        return prompts
    except Exception as e:
        add_log(f"Error updating prompt templates: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

