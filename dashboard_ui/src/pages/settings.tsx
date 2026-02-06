import { useEffect, useMemo, useState } from "react";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { useStore, type DashboardRole } from "@/lib/store";

const roleOptions: DashboardRole[] = ["viewer", "operator", "risk", "admin"];

export default function SettingsPage() {
  const {
    user,
    role,
    mfaStatus,
    mfaEnrollment,
    adminUsers,
    securityStatus,
    ssoStatus,
    siemStatus,
    hasPermission,
    fetchSsoStatus,
    fetchAdminUsers,
    fetchSecurityStatus,
    fetchSiemStatus,
    initMfaEnrollment,
    verifyMfaEnrollment,
    disableMfaEnrollment,
    updateUserRole,
    rotateJwtSecret,
    resetKillSwitch,
    flushSiemQueue,
  } = useStore();

  const [authorizedBy, setAuthorizedBy] = useState(user?.username ?? "operator");
  const [force, setForce] = useState(false);
  const [overrideCode, setOverrideCode] = useState("");
  const [mfaCode, setMfaCode] = useState("");
  const [enrollCode, setEnrollCode] = useState("");
  const [disableCode, setDisableCode] = useState("");
  const [targetUser, setTargetUser] = useState("");
  const [targetRole, setTargetRole] = useState<DashboardRole>("viewer");
  const [rotateSecret, setRotateSecret] = useState("");

  const canResetKillSwitch = hasPermission("control.risk.kill_switch.reset");
  const canManageUsers = hasPermission("admin.users.manage");
  const canRotateSecurity = hasPermission("auth.security.rotate");
  const canReadAudit = hasPermission("control.audit.read");
  const canManageAudit = hasPermission("control.audit.manage");
  const requiresMfa = Boolean(mfaStatus?.mfa_enabled);

  useEffect(() => {
    void fetchSsoStatus();
    if (canManageUsers) {
      void fetchAdminUsers();
    }
    if (canRotateSecurity) {
      void fetchSecurityStatus();
    }
    if (canReadAudit) {
      void fetchSiemStatus();
    }
  }, [canManageUsers, canRotateSecurity, canReadAudit, fetchSsoStatus, fetchAdminUsers, fetchSecurityStatus, fetchSiemStatus]);

  useEffect(() => {
    if (!targetUser && adminUsers.length > 0) {
      setTargetUser(adminUsers[0].username);
      const candidate = adminUsers[0].role as DashboardRole;
      if (roleOptions.includes(candidate)) {
        setTargetRole(candidate);
      }
    }
  }, [adminUsers, targetUser]);

  const selectedAdminUser = useMemo(
    () => adminUsers.find((item) => item.username === targetUser),
    [adminUsers, targetUser],
  );

  return (
    <div className="space-y-6">
      <section className="rounded-2xl border border-slate-200 bg-white/90 p-6">
        <p className="text-xs font-semibold uppercase tracking-[0.2em] text-slate-500">Control Settings</p>
        <h1 className="mt-2 text-3xl font-bold text-slate-900">Security & Governance</h1>
        <p className="mt-1 text-sm text-slate-600">Role access, MFA enrollment, key rotation and privileged controls.</p>
        <div className="mt-3 flex flex-wrap gap-2">
          <Badge variant="outline">Role {role.toUpperCase()}</Badge>
          <Badge variant={mfaStatus?.mfa_enabled ? "success" : "warning"}>
            MFA {mfaStatus?.mfa_enabled ? "ENABLED" : "DISABLED"}
          </Badge>
        </div>
      </section>

      <Card className="border-slate-200 bg-white/90">
        <CardHeader>
          <CardTitle>Kill Switch Reset Controls</CardTitle>
          <CardDescription>Publishes a reset control event to the backend control bus.</CardDescription>
        </CardHeader>
        <CardContent className="space-y-3">
          <label className="block text-sm">
            Authorized By
            <input
              className="mt-1 h-10 w-full rounded-lg border border-slate-300 bg-white px-3"
              value={authorizedBy}
              onChange={(e) => setAuthorizedBy(e.target.value)}
            />
          </label>
          <label className="flex items-center gap-2 text-sm">
            <input type="checkbox" checked={force} onChange={(e) => setForce(e.target.checked)} />
            Force reset (requires override code)
          </label>
          <label className="block text-sm">
            Override Code
            <input
              className="mt-1 h-10 w-full rounded-lg border border-slate-300 bg-white px-3"
              value={overrideCode}
              onChange={(e) => setOverrideCode(e.target.value)}
              placeholder="Required only for force reset"
            />
          </label>
          {requiresMfa ? (
            <label className="block text-sm">
              MFA Code
              <input
                className="mt-1 h-10 w-full rounded-lg border border-slate-300 bg-white px-3 font-mono"
                value={mfaCode}
                onChange={(e) => setMfaCode(e.target.value.replace(/\D/g, "").slice(0, 6))}
                placeholder="6-digit TOTP"
              />
            </label>
          ) : null}
          <Button
            className="bg-slate-900 text-white hover:bg-slate-800"
            onClick={() => void resetKillSwitch(authorizedBy, force, overrideCode || undefined, mfaCode || undefined)}
            disabled={!canResetKillSwitch || (requiresMfa && mfaCode.length !== 6)}
          >
            Publish Reset Command
          </Button>
          {!canResetKillSwitch ? <p className="text-xs text-rose-700">Role lacks kill-switch reset permission.</p> : null}
        </CardContent>
      </Card>

      <Card className="border-slate-200 bg-white/90">
        <CardHeader>
          <CardTitle>MFA Enrollment</CardTitle>
          <CardDescription>Enroll this user with TOTP and enforce MFA on privileged actions.</CardDescription>
        </CardHeader>
        <CardContent className="space-y-3">
          {!mfaStatus?.mfa_enabled ? (
            <>
              <Button variant="outline" onClick={() => void initMfaEnrollment()}>
                Generate Enrollment Secret
              </Button>
              {mfaEnrollment ? (
                <div className="rounded-lg border border-slate-200 bg-slate-50 p-3 text-xs">
                  <p className="font-semibold text-slate-700">Secret: <span className="font-mono">{mfaEnrollment.secret}</span></p>
                  <p className="mt-1 break-all text-slate-600">{mfaEnrollment.provisioning_uri}</p>
                  <label className="mt-3 block text-sm">
                    Verification Code
                    <input
                      className="mt-1 h-10 w-full rounded-lg border border-slate-300 bg-white px-3 font-mono"
                      value={enrollCode}
                      onChange={(e) => setEnrollCode(e.target.value.replace(/\D/g, "").slice(0, 6))}
                      placeholder="6-digit code"
                    />
                  </label>
                  <Button className="mt-2" onClick={() => void verifyMfaEnrollment(enrollCode)} disabled={enrollCode.length !== 6}>
                    Verify & Enable MFA
                  </Button>
                </div>
              ) : null}
            </>
          ) : (
            <>
              <p className="text-sm text-slate-700">MFA is enabled for this account.</p>
              <label className="block text-sm">
                Disable MFA Code
                <input
                  className="mt-1 h-10 w-full rounded-lg border border-slate-300 bg-white px-3 font-mono"
                  value={disableCode}
                  onChange={(e) => setDisableCode(e.target.value.replace(/\D/g, "").slice(0, 6))}
                  placeholder="6-digit code"
                />
              </label>
              <Button variant="outline" onClick={() => void disableMfaEnrollment(disableCode)} disabled={disableCode.length !== 6}>
                Disable MFA
              </Button>
            </>
          )}
        </CardContent>
      </Card>

      {canManageUsers ? (
        <Card className="border-slate-200 bg-white/90">
          <CardHeader>
            <CardTitle>Admin User Role Management</CardTitle>
            <CardDescription>Manage runtime role overrides for dashboard users.</CardDescription>
          </CardHeader>
          <CardContent className="space-y-3">
            <label className="block text-sm">
              User
              <select className="mt-1 h-10 w-full rounded-lg border border-slate-300 bg-white px-3" value={targetUser} onChange={(e) => setTargetUser(e.target.value)}>
                {adminUsers.map((item) => (
                  <option key={item.username} value={item.username}>
                    {item.username}
                  </option>
                ))}
              </select>
            </label>
            <label className="block text-sm">
              Role
              <select className="mt-1 h-10 w-full rounded-lg border border-slate-300 bg-white px-3" value={targetRole} onChange={(e) => setTargetRole(e.target.value as DashboardRole)}>
                {roleOptions.map((item) => (
                  <option key={item} value={item}>
                    {item}
                  </option>
                ))}
              </select>
            </label>
            {requiresMfa ? (
              <label className="block text-sm">
                Admin MFA Code
                <input
                  className="mt-1 h-10 w-full rounded-lg border border-slate-300 bg-white px-3 font-mono"
                  value={mfaCode}
                  onChange={(e) => setMfaCode(e.target.value.replace(/\D/g, "").slice(0, 6))}
                  placeholder="6-digit code"
                />
              </label>
            ) : null}
            <Button
              variant="outline"
              onClick={() => void updateUserRole(targetUser, targetRole, "dashboard_admin_update", mfaCode || undefined)}
              disabled={!targetUser || (requiresMfa && mfaCode.length !== 6)}
            >
              Update Role
            </Button>
            {selectedAdminUser ? (
              <div className="rounded-lg border border-slate-200 bg-slate-50 p-3 text-xs text-slate-600">
                <p>Current role: <span className="font-semibold">{selectedAdminUser.role}</span></p>
                <p>Role source: {selectedAdminUser.role_source}</p>
                <p>MFA enabled: {selectedAdminUser.has_mfa ? "yes" : "no"}</p>
              </div>
            ) : null}
          </CardContent>
        </Card>
      ) : null}

      {canRotateSecurity ? (
        <Card className="border-slate-200 bg-white/90">
          <CardHeader>
            <CardTitle>JWT Key Rotation</CardTitle>
            <CardDescription>Rotate signing key with short key-ring fallback for token continuity.</CardDescription>
          </CardHeader>
          <CardContent className="space-y-3">
            <div className="rounded-lg border border-slate-200 bg-slate-50 p-3 text-xs text-slate-600">
              <p>Active key fingerprint: {securityStatus?.active_key_fingerprint ?? "--"}</p>
              <p>Key ring size: {securityStatus?.jwt_key_count ?? 0}</p>
              <p>Rate limit: {securityStatus?.rate_limit_enabled ? "enabled" : "disabled"} ({securityStatus?.rate_limit_limit ?? 0}/{securityStatus?.rate_limit_window_seconds ?? 0}s)</p>
            </div>
            <label className="block text-sm">
              New JWT Secret
              <input
                className="mt-1 h-10 w-full rounded-lg border border-slate-300 bg-white px-3 font-mono"
                value={rotateSecret}
                onChange={(e) => setRotateSecret(e.target.value)}
                placeholder="at least 32 chars"
              />
            </label>
            {requiresMfa ? (
              <label className="block text-sm">
                MFA Code
                <input
                  className="mt-1 h-10 w-full rounded-lg border border-slate-300 bg-white px-3 font-mono"
                  value={mfaCode}
                  onChange={(e) => setMfaCode(e.target.value.replace(/\D/g, "").slice(0, 6))}
                  placeholder="6-digit code"
                />
              </label>
            ) : null}
            <Button
              variant="outline"
              onClick={() => void rotateJwtSecret(rotateSecret, mfaCode || undefined)}
              disabled={rotateSecret.length < 32 || (requiresMfa && mfaCode.length !== 6)}
            >
              Rotate JWT Secret
            </Button>
          </CardContent>
        </Card>
      ) : null}

      <Card className="border-slate-200 bg-white/90">
        <CardHeader>
          <CardTitle>Enterprise SSO</CardTitle>
          <CardDescription>OIDC federation status for institutional identity providers.</CardDescription>
        </CardHeader>
        <CardContent className="space-y-2 text-sm text-slate-700">
          <p>Enabled: <span className="font-semibold">{ssoStatus?.enabled ? "yes" : "no"}</span></p>
          <p>Configured: <span className="font-semibold">{ssoStatus?.configured ? "yes" : "no"}</span></p>
          <p>Issuer: <span className="font-mono text-xs">{ssoStatus?.issuer ?? "--"}</span></p>
          <p>Role claim: <span className="font-mono text-xs">{ssoStatus?.role_claim ?? "--"}</span></p>
        </CardContent>
      </Card>

      {canReadAudit ? (
        <Card className="border-slate-200 bg-white/90">
          <CardHeader>
            <CardTitle>SIEM Delivery</CardTitle>
            <CardDescription>Forwarding status for centralized audit pipeline.</CardDescription>
          </CardHeader>
          <CardContent className="space-y-2 text-sm text-slate-700">
            <p>Enabled: <span className="font-semibold">{siemStatus?.enabled ? "yes" : "no"}</span></p>
            <p>Endpoint: <span className="font-mono text-xs">{siemStatus?.endpoint ?? "--"}</span></p>
            <p>Queue depth: <span className="font-semibold">{siemStatus?.queue_depth ?? 0}</span></p>
            {siemStatus?.last_error ? <p className="text-xs text-rose-700">{siemStatus.last_error}</p> : null}
            <Button variant="outline" onClick={() => void flushSiemQueue(5)} disabled={!canManageAudit}>
              Flush SIEM Queue
            </Button>
          </CardContent>
        </Card>
      ) : null}
    </div>
  );
}
