"""llm-d guide discovery."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from ..context import ToolContext
from ..registry import tool


class ListGuidesArgs(BaseModel):
    model_config = ConfigDict(extra="forbid")

    cluster_id: str | None = Field(
        default=None, description="Resolve the guide catalog against this cluster's pinned llm-d ref."
    )
    accelerator: str | None = Field(
        default=None, description="Only guides that support this accelerator family, e.g. 'xpu' or 'gpu'."
    )


@tool(
    "list_guides",
    "List the llm-d deployment guides (e.g. optimized-baseline, pd-disaggregation) with the accelerator "
    "families, model servers and variants each supports. Pass a guide id to plan_deployment.",
    ListGuidesArgs,
    read_only=True,
)
async def list_guides(ctx: ToolContext, args: ListGuidesArgs) -> dict[str, Any]:
    catalog = await ctx.api.request("GET", "/guide-planning/catalog", params={"clusterId": args.cluster_id})
    guides = []
    for guide in catalog.get("guides") or []:
        supported = [
            {
                "accelerator": accelerator["id"],
                "model_servers": {
                    server["id"]: server.get("variants") or [] for server in accelerator.get("modelServers") or []
                },
            }
            for accelerator in guide.get("accelerators") or []
            if not args.accelerator or accelerator["id"] == args.accelerator
        ]
        if supported:
            guides.append({"guide": guide["id"], "label": guide.get("label"), "supports": supported})
    return {"ref": catalog.get("ref"), "guides": guides}
