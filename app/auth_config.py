import json
import hmac
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from app.security import register_secret


KAGGLE_ENV_KEYS = ("KAGGLE_USERNAME", "KAGGLE_KEY", "KAGGLE_API_TOKEN", "KAGGLE_CONFIG_DIR")


class AuthConfigError(RuntimeError):
    pass


class AuthSelectionError(ValueError):
    def __init__(self, detail: str, status_code: int = 400):
        super().__init__(detail)
        self.detail = detail
        self.status_code = status_code


@dataclass(frozen=True)
class KaggleCredentials:
    id: str
    username: str = ""
    key: str = field(default="", repr=False)
    api_token: str = field(default="", repr=False)
    config_dir: str = ""

    def apply_to_env(self, env: dict[str, str]) -> None:
        for name in KAGGLE_ENV_KEYS:
            env.pop(name, None)
        if self.config_dir:
            env["KAGGLE_CONFIG_DIR"] = self.config_dir
        if self.username:
            env["KAGGLE_USERNAME"] = self.username
        if self.key:
            env["KAGGLE_KEY"] = self.key
        if self.api_token:
            env["KAGGLE_API_TOKEN"] = self.api_token


@dataclass(frozen=True)
class RelayPrincipal:
    id: str
    allowed_kaggle_key_ids: frozenset[str] | None
    legacy: bool = False

    @property
    def allow_all_keys(self) -> bool:
        return self.allowed_kaggle_key_ids is None

    def allows_key(self, kaggle_key_id: str) -> bool:
        if self.legacy:
            return kaggle_key_id == ""
        if self.allowed_kaggle_key_ids is None:
            return True
        return kaggle_key_id in self.allowed_kaggle_key_ids


class AuthStore:
    def __init__(
        self,
        relay_tokens: list[tuple[str, str, frozenset[str] | None]],
        kaggle_keys: dict[str, KaggleCredentials],
        legacy: bool = False,
    ):
        self.legacy = legacy
        self._kaggle_keys = dict(kaggle_keys)
        self._tokens = [
            (token, RelayPrincipal(token_id, allowed, legacy=legacy))
            for token_id, token, allowed in relay_tokens
        ]

    @classmethod
    def from_settings(cls, settings: Any) -> "AuthStore":
        auth_config_path = getattr(settings, "auth_config_path", None)
        if auth_config_path:
            return cls.from_file(Path(auth_config_path))
        api_token = str(getattr(settings, "api_token", "") or "").strip()
        if not api_token:
            raise AuthConfigError("RELAY_API_TOKEN or RELAY_AUTH_CONFIG is required")
        register_secret(api_token)
        return cls(
            relay_tokens=[("legacy", api_token, frozenset({""}))],
            kaggle_keys={},
            legacy=True,
        )

    @classmethod
    def from_file(cls, path: Path) -> "AuthStore":
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except OSError as exc:
            raise AuthConfigError(f"failed to read RELAY_AUTH_CONFIG: {path}") from exc
        except json.JSONDecodeError as exc:
            raise AuthConfigError(f"RELAY_AUTH_CONFIG is not valid JSON: {path}") from exc
        if not isinstance(data, dict):
            raise AuthConfigError("RELAY_AUTH_CONFIG must be a JSON object")

        kaggle_keys = cls._parse_kaggle_keys(data.get("kaggle_keys"))
        relay_tokens = cls._parse_relay_tokens(data.get("relay_tokens"), set(kaggle_keys))
        return cls(relay_tokens=relay_tokens, kaggle_keys=kaggle_keys)

    @staticmethod
    def _parse_kaggle_keys(raw: Any) -> dict[str, KaggleCredentials]:
        if not isinstance(raw, list) or not raw:
            raise AuthConfigError("RELAY_AUTH_CONFIG requires a non-empty kaggle_keys list")
        result: dict[str, KaggleCredentials] = {}
        for item in raw:
            if not isinstance(item, dict):
                raise AuthConfigError("each kaggle_keys entry must be an object")
            key_id = str(item.get("id", "")).strip()
            if not key_id:
                raise AuthConfigError("each kaggle key requires an id")
            if key_id in result:
                raise AuthConfigError(f"duplicate kaggle key id: {key_id}")
            credentials = KaggleCredentials(
                id=key_id,
                username=str(item.get("username", "") or "").strip(),
                key=str(item.get("key", "") or "").strip(),
                api_token=str(item.get("api_token", "") or "").strip(),
                config_dir=str(item.get("config_dir", "") or "").strip(),
            )
            if not (
                (credentials.username and credentials.key)
                or credentials.api_token
                or credentials.config_dir
            ):
                raise AuthConfigError(
                    f"kaggle key {key_id} requires username/key, api_token, or config_dir"
                )
            result[key_id] = credentials
            register_secret(credentials.key)
            register_secret(credentials.api_token)
        return result

    @staticmethod
    def _parse_relay_tokens(
        raw: Any,
        known_kaggle_key_ids: set[str],
    ) -> list[tuple[str, str, frozenset[str] | None]]:
        if not isinstance(raw, list) or not raw:
            raise AuthConfigError("RELAY_AUTH_CONFIG requires a non-empty relay_tokens list")
        seen_ids: set[str] = set()
        seen_tokens: set[str] = set()
        result: list[tuple[str, str, frozenset[str] | None]] = []
        for item in raw:
            if not isinstance(item, dict):
                raise AuthConfigError("each relay_tokens entry must be an object")
            token_id = str(item.get("id", "")).strip()
            token = str(item.get("token", "") or "").strip()
            if not token_id:
                raise AuthConfigError("each relay token requires an id")
            if not token:
                raise AuthConfigError(f"relay token {token_id} requires a token")
            if token_id in seen_ids:
                raise AuthConfigError(f"duplicate relay token id: {token_id}")
            if token in seen_tokens:
                raise AuthConfigError(f"duplicate relay token value for id: {token_id}")
            allowed_raw = item.get("allowed_kaggle_key_ids")
            if allowed_raw == "*":
                allowed = None
            elif isinstance(allowed_raw, list) and allowed_raw:
                allowed_ids = frozenset(str(value).strip() for value in allowed_raw if str(value).strip())
                if not allowed_ids:
                    raise AuthConfigError(f"relay token {token_id} has no allowed kaggle keys")
                unknown = sorted(allowed_ids - known_kaggle_key_ids)
                if unknown:
                    raise AuthConfigError(
                        f"relay token {token_id} references unknown kaggle keys: {', '.join(unknown)}"
                    )
                allowed = allowed_ids
            else:
                raise AuthConfigError(
                    f"relay token {token_id} requires allowed_kaggle_key_ids as '*' or a non-empty list"
                )
            seen_ids.add(token_id)
            seen_tokens.add(token)
            register_secret(token)
            result.append((token_id, token, allowed))
        return result

    def authenticate_authorization(self, authorization: str) -> RelayPrincipal | None:
        token = bearer_token(authorization)
        if not token:
            return None
        return self.authenticate_token(token)

    def authenticate_token(self, token: str) -> RelayPrincipal | None:
        if not token:
            return None
        for expected, principal in self._tokens:
            if hmac.compare_digest(token, expected):
                return principal
        return None

    def allowed_key_ids(self, principal: RelayPrincipal) -> list[str]:
        if self.legacy:
            return [""]
        if principal.allowed_kaggle_key_ids is None:
            return sorted(self._kaggle_keys)
        return sorted(principal.allowed_kaggle_key_ids)

    def resolve_kaggle_key_id(self, principal: RelayPrincipal, requested: str | None) -> str:
        requested_key_id = str(requested or "").strip()
        if self.legacy:
            if requested_key_id:
                raise AuthSelectionError("kaggle_key_id is not available in legacy mode", 400)
            return ""

        if requested_key_id:
            if requested_key_id not in self._kaggle_keys:
                raise AuthSelectionError("unknown kaggle_key_id", 400)
            if not principal.allows_key(requested_key_id):
                raise AuthSelectionError("kaggle_key_id is not allowed for this token", 403)
            return requested_key_id

        allowed = self.allowed_key_ids(principal)
        if len(allowed) == 1:
            return allowed[0]
        raise AuthSelectionError("kaggle_key_id is required for this token", 400)

    def credentials_for(self, kaggle_key_id: str) -> KaggleCredentials | None:
        key_id = str(kaggle_key_id or "").strip()
        if not key_id:
            return None
        credentials = self._kaggle_keys.get(key_id)
        if not credentials:
            raise AuthSelectionError("unknown kaggle_key_id", 400)
        return credentials

    def can_access_key(self, principal: RelayPrincipal, kaggle_key_id: str) -> bool:
        return principal.allows_key(str(kaggle_key_id or "").strip())


def bearer_token(authorization: str) -> str:
    prefix = "Bearer "
    value = str(authorization or "")
    if not value.startswith(prefix):
        return ""
    return value[len(prefix):].strip()
