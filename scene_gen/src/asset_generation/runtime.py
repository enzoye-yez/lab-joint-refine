"""Deterministic contact events and connection release; the model may not write success states directly."""

import math
import numpy as np

class InteractionRuntime:
    def __init__(self, model, manifest, actor_geoms=(), *, actor_geom_ids=()):
        self.model = model
        self.buttons = manifest.get("buttons", [])
        self.connections = manifest.get("connections", [])
        self.unverified_mechanical_events = list(manifest.get("mechanical_events", []))
        self.manifest_interfaces = manifest["interfaces"]
        if any(type(g) is not int or not 0 <= g < model.ngeom for g in actor_geom_ids):
            raise ValueError("Contact object geom id does not belong to the current model")
        self.actor_geoms = {model.geom(name).id for name in actor_geoms} | set(actor_geom_ids)
        self.pressed = {b["id"]: False for b in self.buttons}
        self.released = set()

    def update(self, data, software_requests=(), *, verified_mechanical_events=()):
        software = {i["id"] for i in self.manifest_interfaces if i["kind"] == "software"}
        if any(event not in software for event in software_requests):
            raise ValueError("Undeclared software control event")
        events = list(dict.fromkeys(software_requests))
        # Only for the fixed operation detectors; model software requests still cannot trigger mechanical events.
        mechanical = {e["id"] for e in self.unverified_mechanical_events}
        if any(event not in mechanical for event in verified_mechanical_events):
            raise ValueError("Undeclared mechanical detection event")
        events.extend(event for event in verified_mechanical_events if event not in events)
        for button in self.buttons:
            joint = self.model.joint(button["joint"])
            position = float(data.qpos[joint.qposadr[0]])
            geom_id = self.model.geom(button["geom"]).id
            touching = any(c.dist <= 0 and (
                (int(c.geom1) == geom_id and int(c.geom2) in self.actor_geoms) or
                (int(c.geom2) == geom_id and int(c.geom1) in self.actor_geoms)) for c in data.contact)
            if self.pressed[button["id"]]:
                if position <= button["released_at"]:
                    self.pressed[button["id"]] = False
            elif touching and position >= button["pressed_at"]:
                self.pressed[button["id"]] = True
                events.append(button["event"])
        records = [{"type": "event", "name": event, "time": float(data.time)} for event in events]
        for connection in self.connections:
            if connection["release_event"] not in events or connection["id"] in self.released:
                continue
            eq_id = self.model.equality(connection["equality"]).id
            if data.eq_active[eq_id]:
                data.eq_active[eq_id] = False
                self.released.add(connection["id"])
                records.append({"type": "connection_released", "id": connection["id"],
                                "irreversible": connection["irreversible"], "time": float(data.time)})
        for connection in self.connections:
            if not connection.get("attach_event") or connection["attach_event"] not in events:
                continue
            eq_id = self.model.equality(connection["equality"]).id
            if connection["irreversible"] or data.eq_active[eq_id]:
                continue
            s1,s2 = [self.model.site(n).id for n in connection["capture_sites"]]
            separation = float(np.linalg.norm(data.site_xpos[s1]-data.site_xpos[s2]))
            rotation = data.site_xmat[s1].reshape(3,3).T @ data.site_xmat[s2].reshape(3,3)
            angle = math.acos(float(np.clip((np.trace(rotation)-1)/2,-1,1)))
            if separation <= connection["position_tolerance_m"] and angle <= connection["angle_tolerance_rad"]:
                data.eq_active[eq_id] = True
                self.released.discard(connection["id"])
                records.append({"type":"connection_attached","id":connection["id"],"time":float(data.time),
                                "separation_m":separation,"angle_rad":angle})
        return records

