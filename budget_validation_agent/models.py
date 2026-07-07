# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


from pydantic import BaseModel, Field


class PurchaseRequest(BaseModel):
    request_id: str = Field(description="Unique identifier for the purchase request")
    department: str = Field(description="Department making the request")
    project: str = Field(
        description="Project name or code associated with the purchase"
    )
    requested_amount: float = Field(description="Requested amount for the purchase")
    available_budget: float = Field(description="Available budget for the project")
    requester: str = Field(description="Name of the person requesting the purchase")
    description: str = Field(description="Description of the purchase item(s)")
    date: str = Field(description="Date of the request (YYYY-MM-DD)")


class RiskAssessment(BaseModel):
    risk_level: str = Field(
        description="Calculated budget risk level: low, medium, or high"
    )
    risk_factors: list[str] = Field(
        description="Key factors contributing to the risk level"
    )
    alert_raised: bool = Field(description="Whether a budget risk alert is raised")
    justification: str = Field(
        description="Reasoning/justification for the risk assessment"
    )


class ValidationResult(BaseModel):
    approved: bool = Field(description="Whether the purchase request is approved")
    status: str = Field(
        description="Final status of the request: auto_approved, approved, or rejected"
    )
    reason: str = Field(description="Explanation of the final validation outcome")
