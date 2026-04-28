"""Registry of REPL agent modes."""

from __future__ import annotations

from euler_agent.agent_modes.assistant import SPEC as ASSISTANT_SPEC
from euler_agent.agent_modes.architect import SPEC as ARCHITECT_SPEC
from euler_agent.agent_modes.basic import SPEC as BASIC_SPEC
from euler_agent.agent_modes.coder import SPEC as CODER_SPEC
from euler_agent.agent_modes.db import SPEC as DB_SPEC
from euler_agent.agent_modes.devops import SPEC as DEVOPS_SPEC
from euler_agent.agent_modes.documenter import SPEC as DOCUMENTER_SPEC
from euler_agent.agent_modes.patch import SPEC as PATCH_SPEC
from euler_agent.agent_modes.refactor import SPEC as REFACTOR_SPEC
from euler_agent.agent_modes.security import SPEC as SECURITY_SPEC
from euler_agent.agent_modes.swarm import SPEC as SWARM_SPEC
from euler_agent.agent_modes.tester import SPEC as TESTER_SPEC
from euler_agent.agent_modes.types import AgentModeSpec

MODE_SPECS: tuple[AgentModeSpec, ...] = (
    BASIC_SPEC,
    ASSISTANT_SPEC,
    SWARM_SPEC,
    PATCH_SPEC,
    ARCHITECT_SPEC,
    CODER_SPEC,
    TESTER_SPEC,
    SECURITY_SPEC,
    DEVOPS_SPEC,
    DB_SPEC,
    DOCUMENTER_SPEC,
    REFACTOR_SPEC,
)

MODE_BY_NAME: dict[str, AgentModeSpec] = {spec.name: spec for spec in MODE_SPECS}
