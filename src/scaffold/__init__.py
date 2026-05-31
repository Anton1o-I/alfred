"""scaffold — reusable infra for agent systems.

This package is the substrate Alfred is built on: model routing, storage,
audit logging, budget enforcement, notification channels, scheduling,
and the tool registry. It MUST NOT import from any specific agent
package (e.g. `alfred`). Agent systems depend on `scaffold`; never the
other way around.
"""
