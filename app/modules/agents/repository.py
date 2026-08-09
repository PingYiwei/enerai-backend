from __future__ import annotations

import base64
from datetime import UTC, datetime
from typing import Any

from gridfs import AsyncGridFSBucket
from pymongo import ASCENDING, DESCENDING, ReturnDocument
from pymongo.asynchronous.database import AsyncDatabase

from app.core.errors import AppError
from app.core.ids import new_id
from app.core.security import Principal
from app.modules.agents.context import contextual_content, validate_references
from app.modules.agents.schemas import (
    RunAccepted,
    RunCreate,
    RunStatus,
    SessionCreate,
    SessionEntry,
    SessionList,
    SessionSnapshot,
    SessionSummary,
    SessionUpdate,
)
from app.modules.agents.types import ImageInput, JsonObject, Message, ToolCall, Usage

Document = dict[str, Any]


class MongoAgentRepository:
    def __init__(self, database: AsyncDatabase[Document]) -> None:
        self._database = database

    @property
    def database(self) -> AsyncDatabase[Document]:
        return self._database

    async def create_session(
        self,
        principal: Principal,
        project_id: str,
        request: SessionCreate,
        surface: str = "insight",
    ) -> SessionSummary:
        project = await self._database.projects.find_one(
            {"_id": project_id, "owner_id": principal.user_id}, {"_id": 1}
        )
        if project is None:
            raise AppError("project_not_found", "Project was not found", status_code=404)
        now = datetime.now(UTC)
        session_id = new_id("ses")
        document = {
            "_id": session_id,
            "owner_id": principal.user_id,
            "project_id": project_id,
            "title": request.title.strip(),
            "surface": surface,
            "status": "active",
            "sequence": 0,
            "revision": 0,
            "created_at": now,
            "updated_at": now,
        }
        await self._database.agent_sessions.insert_one(document)
        await self._database.agent_lanes.update_one(
            {"session_id": session_id, "name": "main"},
            {
                "$setOnInsert": {
                    "_id": f"{session_id}:main",
                    "session_id": session_id,
                    "name": "main",
                    "leaf_id": None,
                    "revision": 0,
                    "active_operation": None,
                }
            },
            upsert=True,
        )
        return _session_summary(document)

    async def list_sessions(
        self, principal: Principal, project_id: str, surface: str = "insight"
    ) -> SessionList:
        documents = await (
            self._database.agent_sessions.find(
                {
                    "owner_id": principal.user_id,
                    "project_id": project_id,
                    "surface": surface,
                }
            )
            .sort("updated_at", DESCENDING)
            .to_list(None)
        )
        return SessionList(
            items=[_session_summary(document) for document in documents],
            total=len(documents),
        )

    async def update_session(
        self, principal: Principal, session_id: str, request: SessionUpdate
    ) -> SessionSummary:
        title = request.title.strip()
        if not title:
            raise AppError(
                "invalid_session_title", "Session title cannot be empty", status_code=422
            )
        document = await self._database.agent_sessions.find_one_and_update(
            {"_id": session_id, "owner_id": principal.user_id},
            {"$set": {"title": title, "updated_at": datetime.now(UTC)}},
            return_document=ReturnDocument.AFTER,
        )
        if document is None:
            raise AppError("session_not_found", "Session was not found", status_code=404)
        return _session_summary(document)

    async def automatic_title_target(
        self, principal: Principal, session_id: str
    ) -> str | None:
        session = await self._owned_session(principal, session_id)
        title = str(session.get("title") or "").strip()
        if session.get("surface", "insight") != "insight":
            return None
        return title if title.casefold() in {"new insight", "new session"} else None

    async def apply_generated_title(
        self,
        session_id: str,
        owner_id: str,
        expected_title: str,
        title: str,
    ) -> bool:
        result = await self._database.agent_sessions.update_one(
            {"_id": session_id, "owner_id": owner_id, "title": expected_title},
            {
                "$set": {
                    "title": title,
                    "title_generated_at": datetime.now(UTC),
                    "updated_at": datetime.now(UTC),
                }
            },
        )
        return result.modified_count == 1

    async def delete_session(self, principal: Principal, session_id: str) -> None:
        await self._owned_session(principal, session_id)
        active = await self._database.agent_operations.find_one(
            {
                "session_id": session_id,
                "owner_id": principal.user_id,
                "status": {"$in": ["accepted", "running"]},
            },
            {"_id": 1},
        )
        if active is not None:
            raise AppError(
                "session_busy",
                "Stop the active run before deleting this session",
                status_code=409,
            )

        operations = await self._database.agent_operations.find(
            {"session_id": session_id, "owner_id": principal.user_id}, {"_id": 1}
        ).to_list(None)
        run_ids = [document["_id"] for document in operations]
        artifacts = await self._database.artifacts.find(
            {"session_id": session_id, "owner_id": principal.user_id}, {"file_id": 1}
        ).to_list(None)
        attachments = await self._database.chat_attachments.find(
            {"session_id": session_id, "owner_id": principal.user_id}, {"file_id": 1}
        ).to_list(None)

        artifact_bucket = AsyncGridFSBucket(self._database, bucket_name="agent_artifacts")
        for artifact in artifacts:
            await artifact_bucket.delete(artifact["file_id"])
        attachment_bucket = AsyncGridFSBucket(self._database, bucket_name="chat_files")
        for attachment in attachments:
            await attachment_bucket.delete(attachment["file_id"])

        if run_ids:
            await self._database.agent_events.delete_many({"run_id": {"$in": run_ids}})
        await self._database.agent_records.delete_many({"session_id": session_id})
        await self._database.agent_entries.delete_many({"session_id": session_id})
        await self._database.agent_lanes.delete_many({"session_id": session_id})
        await self._database.agent_operations.delete_many({"session_id": session_id})
        await self._database.artifacts.delete_many({"session_id": session_id})
        await self._database.chat_attachments.delete_many({"session_id": session_id})
        await self._database.agent_sessions.delete_one(
            {"_id": session_id, "owner_id": principal.user_id}
        )

    async def snapshot(self, principal: Principal, session_id: str) -> SessionSnapshot:
        session = await self._owned_session(principal, session_id)
        lane = await self._ensure_lane(session_id)
        documents = await (
            self._database.agent_entries.find({"session_id": session_id})
            .sort("seq", ASCENDING)
            .to_list(None)
        )
        by_id = {document["_id"]: document for document in documents}
        path: list[Document] = []
        entry_id = lane.get("leaf_id")
        while entry_id is not None:
            entry = by_id.get(entry_id)
            if entry is None:
                raise AppError(
                    "session_corrupt",
                    "Session lane references a missing entry",
                    status_code=500,
                )
            path.append(entry)
            entry_id = entry.get("parent_id")
        path.reverse()
        run_ids = list({str(document["run_id"]) for document in path if document.get("run_id")})
        usage_by_run: dict[str, Document] = {}
        if run_ids:
            operations = await self._database.agent_operations.find(
                {"_id": {"$in": run_ids}}, {"usage": 1}
            ).to_list(None)
            usage_by_run = {
                str(operation["_id"]): operation["usage"]
                for operation in operations
                if isinstance(operation.get("usage"), dict)
            }
        active = lane.get("active_operation")
        return SessionSnapshot(
            session=_session_summary(session),
            revision=int(session.get("revision", 0)),
            lane="main",
            leaf_id=lane.get("leaf_id"),
            active_run_id=active.get("id") if isinstance(active, dict) else None,
            entries=[
                _session_entry(document, usage_by_run.get(str(document.get("run_id", ""))))
                for document in path
            ],
        )

    async def accept_run(
        self,
        principal: Principal,
        session_id: str,
        request: RunCreate,
        provider: str,
        model: str,
    ) -> RunAccepted:
        session = await self._owned_session(principal, session_id)
        lane = await self._ensure_lane(session_id)
        attachments = await self._draft_attachments(
            principal, session["project_id"], request.attachment_ids
        )
        references = await validate_references(
            self._database, principal, session["project_id"], request.references
        )
        now = datetime.now(UTC)
        run_id = new_id("run")
        user_entry_id = f"ent_{run_id.removeprefix('run_')}_0000"
        source_leaf_id = str(lane["leaf_id"]) if lane.get("leaf_id") is not None else None
        intent: Document = {
            "id": run_id,
            "source_leaf_id": source_leaf_id,
            "user_entry_id": user_entry_id,
            "content": request.content,
            "provider": provider,
            "api_style": request.api_style,
            "model": model,
            "attachment_ids": request.attachment_ids,
            "references": [reference.model_dump() for reference in references],
            "accepted_at": now,
        }
        claimed = await self._database.agent_lanes.find_one_and_update(
            {
                "_id": lane["_id"],
                "revision": lane["revision"],
                "active_operation": None,
            },
            {"$set": {"active_operation": intent}, "$inc": {"revision": 1}},
            return_document=ReturnDocument.AFTER,
        )
        if claimed is None:
            raise AppError(
                "lane_busy",
                "This session already has an active run",
                status_code=409,
            )
        await self._database.agent_operations.update_one(
            {"_id": run_id},
            {
                "$setOnInsert": {
                    "_id": run_id,
                    "session_id": session_id,
                    "project_id": session["project_id"],
                    "surface": session.get("surface", "insight"),
                    "owner_id": principal.user_id,
                    "lane": "main",
                    "status": "accepted",
                    "provider": provider,
                    "api_style": request.api_style,
                    "model": model,
                    "event_sequence": 0,
                    "record_sequence": 1,
                    "persisted_message_count": 1,
                    "created_at": now,
                    "updated_at": now,
                    "error": None,
                }
            },
            upsert=True,
        )
        if attachments:
            await self._database.chat_attachments.update_many(
                {"_id": {"$in": request.attachment_ids}, "status": "draft"},
                {
                    "$set": {
                        "status": "committed",
                        "session_id": session_id,
                        "run_id": run_id,
                    }
                },
            )
        await self._database.agent_records.update_one(
            {"_id": f"{run_id}:1"},
            {
                "$setOnInsert": {
                    "_id": f"{run_id}:1",
                    "operation_id": run_id,
                    "session_id": session_id,
                    "seq": 1,
                    "type": "run_intent",
                    "data": {
                        "source_leaf_id": source_leaf_id,
                        "user_entry_id": user_entry_id,
                        "provider": provider,
                        "model": model,
                    },
                    "created_at": now,
                }
            },
            upsert=True,
        )
        await self._append_entry(
            session_id=session_id,
            entry_id=user_entry_id,
            parent_id=intent["source_leaf_id"],
            message=Message(
                role="user",
                content=request.content,
                images=tuple(
                    ImageInput(
                        id=document["_id"],
                        name=document["name"],
                        media_type=document["media_type"],
                    )
                    for document in attachments
                ),
            ),
            run_id=run_id,
            references=[reference.model_dump() for reference in references],
        )
        await self._database.agent_lanes.update_one(
            {"_id": lane["_id"], "active_operation.id": run_id},
            {"$set": {"leaf_id": user_entry_id}},
        )
        await self._touch_session(session_id)
        return RunAccepted(id=run_id, session_id=session_id, status="accepted", accepted_at=now)

    async def run_context(self, run_id: str) -> tuple[Document, list[Message]]:
        operation = await self._database.agent_operations.find_one({"_id": run_id})
        if operation is None:
            raise AppError("run_not_found", "Run was not found", status_code=404)
        snapshot = await self.snapshot(
            Principal(user_id=operation["owner_id"], username=""), operation["session_id"]
        )
        messages: list[Message] = []
        for entry in snapshot.entries:
            images = await self._load_images(entry.attachments)
            messages.append(
                Message(
                    role=entry.role,
                    content=(
                        contextual_content(entry.content, entry.references)
                        if entry.role == "user"
                        else entry.content
                    ),
                    tool_calls=tuple(
                        ToolCall(
                            id=str(call["id"]),
                            name=str(call["name"]),
                            arguments=str(call["arguments"]),
                        )
                        for call in entry.tool_calls
                    ),
                    tool_call_id=entry.tool_call_id,
                    images=images,
                )
            )
        return operation, messages

    async def mark_running(self, run_id: str) -> bool:
        operation = await self._database.agent_operations.find_one_and_update(
            {"_id": run_id, "status": "accepted"},
            {"$set": {"status": "running", "updated_at": datetime.now(UTC)}},
            return_document=ReturnDocument.AFTER,
        )
        return operation is not None

    async def recover_incomplete_runs(self) -> tuple[list[Document], list[str]]:
        """Return safe-to-restart runs and reconcile runs interrupted mid-effect.

        An accepted operation has persisted its user input but has not claimed the
        provider step, so it is safe to schedule again. A running operation may
        have partially executed a provider request or a write tool. Replaying that
        work would violate at-most-once semantics, so it is failed explicitly and
        its lane is released.
        """
        await self._reconcile_orphan_intents()
        incomplete = await (
            self._database.agent_operations.find({"status": {"$in": ["accepted", "running"]}})
            .sort("created_at", ASCENDING)
            .to_list(None)
        )
        accepted: list[Document] = []
        interrupted: list[str] = []
        for operation in incomplete:
            if operation["status"] == "accepted":
                accepted.append(operation)
                continue
            run_id = str(operation["_id"])
            await self.append_event(
                run_id,
                "run_end",
                {
                    "run_id": run_id,
                    "outcome": "failed",
                    "error": "server_restarted_during_effect",
                },
            )
            await self.finish_run(
                run_id,
                "failed",
                error="Run was interrupted by a server restart during an external effect",
            )
            interrupted.append(run_id)
        return accepted, interrupted

    async def _reconcile_orphan_intents(self) -> None:
        lanes = await self._database.agent_lanes.find({"active_operation": {"$ne": None}}).to_list(
            None
        )
        for lane in lanes:
            intent = lane.get("active_operation")
            if not isinstance(intent, dict) or not intent.get("id"):
                continue
            run_id = str(intent["id"])
            session = await self._database.agent_sessions.find_one({"_id": lane["session_id"]})
            if session is None:
                continue
            accepted_at = intent.get("accepted_at") or datetime.now(UTC)
            result = await self._database.agent_operations.update_one(
                {"_id": run_id},
                {
                    "$setOnInsert": {
                        "_id": run_id,
                        "session_id": session["_id"],
                        "project_id": session["project_id"],
                        "surface": session.get("surface", "insight"),
                        "owner_id": session["owner_id"],
                        "lane": lane.get("name", "main"),
                        "status": "accepted",
                        "provider": intent["provider"],
                        "api_style": intent.get("api_style"),
                        "model": intent["model"],
                        "event_sequence": 0,
                        "record_sequence": 1,
                        "persisted_message_count": 1,
                        "created_at": accepted_at,
                        "updated_at": accepted_at,
                        "error": None,
                    }
                },
                upsert=True,
            )
            if result.upserted_id is None:
                continue
            attachment_ids = [str(value) for value in intent.get("attachment_ids", [])]
            attachments = await self._database.chat_attachments.find(
                {
                    "_id": {"$in": attachment_ids},
                    "owner_id": session["owner_id"],
                    "project_id": session["project_id"],
                }
            ).to_list(None)
            await self._database.chat_attachments.update_many(
                {"_id": {"$in": attachment_ids}},
                {
                    "$set": {
                        "status": "committed",
                        "session_id": session["_id"],
                        "run_id": run_id,
                    }
                },
            )
            await self._database.agent_records.insert_one(
                {
                    "_id": f"{run_id}:1",
                    "operation_id": run_id,
                    "session_id": session["_id"],
                    "seq": 1,
                    "type": "run_intent",
                    "data": {"recovered": True},
                    "created_at": accepted_at,
                }
            )
            await self._append_entry(
                session_id=session["_id"],
                entry_id=str(intent["user_entry_id"]),
                parent_id=intent.get("source_leaf_id"),
                message=Message(
                    role="user",
                    content=str(intent["content"]),
                    images=tuple(
                        ImageInput(
                            id=document["_id"],
                            name=document["name"],
                            media_type=document["media_type"],
                        )
                        for document in attachments
                    ),
                ),
                run_id=run_id,
                references=list(intent.get("references", [])),
            )
            await self._database.agent_lanes.update_one(
                {"_id": lane["_id"], "active_operation.id": run_id},
                {"$set": {"leaf_id": intent["user_entry_id"]}},
            )

    async def append_event(self, run_id: str, event_type: str, data: JsonObject) -> None:
        operation = await self._database.agent_operations.find_one_and_update(
            {"_id": run_id},
            {"$inc": {"event_sequence": 1}, "$set": {"updated_at": datetime.now(UTC)}},
            return_document=ReturnDocument.AFTER,
        )
        if operation is None:
            raise AppError("run_not_found", "Run was not found", status_code=404)
        await self._database.agent_events.insert_one(
            {
                "_id": f"{run_id}:{operation['event_sequence']}",
                "run_id": run_id,
                "session_id": operation["session_id"],
                "seq": operation["event_sequence"],
                "type": event_type,
                "data": data,
                "created_at": datetime.now(UTC),
            }
        )
        record_type = {
            "message_start": "provider_intent",
            "message_end": "provider_result",
            "tool_execution_start": "tool_intent",
            "tool_execution_end": "tool_result",
            "run_end": "run_result",
        }.get(event_type)
        if record_type is not None:
            await self._append_record(run_id, record_type, data)

    async def persist_checkpoint(
        self,
        run_id: str,
        messages: tuple[Message, ...],
        initial_count: int,
    ) -> None:
        operation = await self._database.agent_operations.find_one({"_id": run_id})
        if operation is None:
            raise AppError("run_not_found", "Run was not found", status_code=404)
        persisted = int(operation.get("persisted_message_count", 1))
        run_messages = messages[initial_count:]
        parent_id = await self._run_leaf(operation)
        for run_index, message in enumerate(run_messages, start=1):
            if run_index < persisted:
                continue
            entry_id = f"ent_{run_id.removeprefix('run_')}_{run_index:04d}"
            await self._append_entry(
                session_id=operation["session_id"],
                entry_id=entry_id,
                parent_id=parent_id,
                message=message,
                run_id=run_id,
            )
            parent_id = entry_id
            await self._database.agent_operations.update_one(
                {"_id": run_id},
                {
                    "$set": {
                        "persisted_message_count": run_index + 1,
                        "last_entry_id": entry_id,
                        "updated_at": datetime.now(UTC),
                    }
                },
            )
            await self._database.agent_lanes.update_one(
                {
                    "session_id": operation["session_id"],
                    "name": "main",
                    "active_operation.id": run_id,
                },
                {"$set": {"leaf_id": entry_id}},
            )

    async def finish_run(
        self,
        run_id: str,
        status: str,
        usage: Usage | None = None,
        error: str | None = None,
    ) -> None:
        now = datetime.now(UTC)
        operation = await self._database.agent_operations.find_one_and_update(
            {"_id": run_id},
            {
                "$set": {
                    "status": status,
                    "usage": (
                        {
                            "input_tokens": usage.input_tokens,
                            "output_tokens": usage.output_tokens,
                            "cached_input_tokens": usage.cached_input_tokens,
                            "reasoning_tokens": usage.reasoning_tokens,
                        }
                        if usage is not None
                        else None
                    ),
                    "error": error,
                    "updated_at": now,
                    "completed_at": now,
                }
            },
            return_document=ReturnDocument.AFTER,
        )
        if operation is None:
            return
        await self._database.agent_lanes.update_one(
            {"session_id": operation["session_id"], "name": "main", "active_operation.id": run_id},
            {"$set": {"active_operation": None}},
        )
        await self._touch_session(operation["session_id"])

    async def get_run(self, principal: Principal, run_id: str) -> RunStatus:
        operation = await self._database.agent_operations.find_one(
            {"_id": run_id, "owner_id": principal.user_id}
        )
        if operation is None:
            raise AppError("run_not_found", "Run was not found", status_code=404)
        return _run_status(operation)

    async def list_events(
        self, principal: Principal, run_id: str, after: int
    ) -> tuple[RunStatus, list[Document]]:
        run = await self.get_run(principal, run_id)
        events = await (
            self._database.agent_events.find({"run_id": run_id, "seq": {"$gt": after}})
            .sort("seq", ASCENDING)
            .to_list(None)
        )
        return run, events

    async def _owned_session(self, principal: Principal, session_id: str) -> Document:
        session = await self._database.agent_sessions.find_one(
            {"_id": session_id, "owner_id": principal.user_id}
        )
        if session is None:
            raise AppError("session_not_found", "Session was not found", status_code=404)
        return session

    async def _append_record(self, run_id: str, record_type: str, data: JsonObject) -> None:
        operation = await self._database.agent_operations.find_one_and_update(
            {"_id": run_id},
            {"$inc": {"record_sequence": 1}},
            return_document=ReturnDocument.AFTER,
        )
        if operation is None:
            raise AppError("run_not_found", "Run was not found", status_code=404)
        sequence = int(operation["record_sequence"])
        await self._database.agent_records.insert_one(
            {
                "_id": f"{run_id}:{sequence}",
                "operation_id": run_id,
                "session_id": operation["session_id"],
                "seq": sequence,
                "type": record_type,
                "data": data,
                "created_at": datetime.now(UTC),
            }
        )

    async def _ensure_lane(self, session_id: str) -> Document:
        lane = await self._database.agent_lanes.find_one({"session_id": session_id, "name": "main"})
        if lane is not None:
            return lane
        await self._database.agent_lanes.update_one(
            {"session_id": session_id, "name": "main"},
            {
                "$setOnInsert": {
                    "_id": f"{session_id}:main",
                    "session_id": session_id,
                    "name": "main",
                    "leaf_id": None,
                    "revision": 0,
                    "active_operation": None,
                }
            },
            upsert=True,
        )
        lane = await self._database.agent_lanes.find_one({"session_id": session_id, "name": "main"})
        if lane is None:
            raise AppError("session_corrupt", "Session lane could not be created", status_code=500)
        return lane

    async def _append_entry(
        self,
        *,
        session_id: str,
        entry_id: str,
        parent_id: str | None,
        message: Message,
        run_id: str,
        references: list[dict[str, Any]] | None = None,
    ) -> None:
        existing = await self._database.agent_entries.find_one({"_id": entry_id})
        if existing is not None:
            return
        session = await self._database.agent_sessions.find_one_and_update(
            {"_id": session_id},
            {"$inc": {"sequence": 1, "revision": 1}},
            return_document=ReturnDocument.AFTER,
        )
        if session is None:
            raise AppError("session_not_found", "Session was not found", status_code=404)
        await self._database.agent_entries.insert_one(
            {
                "_id": entry_id,
                "session_id": session_id,
                "run_id": run_id,
                "seq": session["sequence"],
                "parent_id": parent_id,
                "role": message.role,
                "content": message.content,
                "tool_calls": [
                    {"id": call.id, "name": call.name, "arguments": call.arguments}
                    for call in message.tool_calls
                ],
                "tool_call_id": message.tool_call_id,
                "attachments": [
                    {
                        "id": image.id,
                        "name": image.name,
                        "media_type": image.media_type,
                    }
                    for image in message.images
                ],
                "references": references or [],
                "created_at": datetime.now(UTC),
            }
        )

    async def _draft_attachments(
        self, principal: Principal, project_id: str, attachment_ids: list[str]
    ) -> list[Document]:
        if not attachment_ids:
            return []
        if len(attachment_ids) != len(set(attachment_ids)):
            raise AppError("duplicate_attachment", "Attachment IDs must be unique", status_code=422)
        documents = await self._database.chat_attachments.find(
            {
                "_id": {"$in": attachment_ids},
                "owner_id": principal.user_id,
                "project_id": project_id,
                "status": "draft",
            }
        ).to_list(None)
        by_id = {document["_id"]: document for document in documents}
        if len(by_id) != len(attachment_ids):
            raise AppError(
                "attachment_not_available",
                "An attachment is missing, already used, or belongs to another project",
                status_code=409,
            )
        ordered = [by_id[attachment_id] for attachment_id in attachment_ids]
        if sum(int(document["size"]) for document in ordered) > 20 * 1024 * 1024:
            raise AppError(
                "attachments_too_large",
                "Combined image attachments exceed the 20 MB run limit",
                status_code=413,
            )
        return ordered

    async def _load_images(self, attachments: list[dict[str, Any]]) -> tuple[ImageInput, ...]:
        if not attachments:
            return ()
        ids = [attachment["id"] for attachment in attachments]
        documents = await self._database.chat_attachments.find({"_id": {"$in": ids}}).to_list(None)
        by_id = {document["_id"]: document for document in documents}
        bucket = AsyncGridFSBucket(self._database, bucket_name="chat_files")
        images: list[ImageInput] = []
        for attachment in attachments:
            document = by_id.get(attachment["id"])
            if document is None:
                raise AppError(
                    "attachment_missing",
                    "A session image is no longer available",
                    status_code=500,
                )
            stream = await bucket.open_download_stream(document["file_id"])
            content = await stream.read()
            images.append(
                ImageInput(
                    id=document["_id"],
                    name=document["name"],
                    media_type=document["media_type"],
                    data_base64=base64.b64encode(content).decode(),
                )
            )
        return tuple(images)

    async def _run_leaf(self, operation: Document) -> str | None:
        if operation.get("last_entry_id"):
            return str(operation["last_entry_id"])
        lane = await self._database.agent_lanes.find_one(
            {"session_id": operation["session_id"], "name": "main"}
        )
        return lane.get("leaf_id") if lane is not None else None

    async def _touch_session(self, session_id: str) -> None:
        await self._database.agent_sessions.update_one(
            {"_id": session_id}, {"$set": {"updated_at": datetime.now(UTC)}}
        )


def _session_summary(document: Document) -> SessionSummary:
    return SessionSummary(
        id=document["_id"],
        project_id=document["project_id"],
        title=document["title"],
        surface=document.get("surface", "insight"),
        status=document.get("status", "active"),
        created_at=document["created_at"],
        updated_at=document["updated_at"],
    )


def _session_entry(document: Document, usage: Document | None = None) -> SessionEntry:
    return SessionEntry(
        id=document["_id"],
        seq=document["seq"],
        parent_id=document.get("parent_id"),
        role=document["role"],
        content=document.get("content", ""),
        run_id=document.get("run_id"),
        usage=usage,
        tool_calls=document.get("tool_calls", []),
        tool_call_id=document.get("tool_call_id"),
        attachments=document.get("attachments", []),
        references=document.get("references", []),
        created_at=document["created_at"],
    )


def _run_status(document: Document) -> RunStatus:
    return RunStatus(
        id=document["_id"],
        session_id=document["session_id"],
        status=document["status"],
        error=document.get("error"),
        created_at=document["created_at"],
        updated_at=document["updated_at"],
    )
