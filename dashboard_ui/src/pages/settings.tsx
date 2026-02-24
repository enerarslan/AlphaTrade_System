import { useEffect, useMemo, useState } from "react";
import { motion } from "framer-motion";
import { useShallow } from "zustand/react/shallow";
import { Globe, Key, Lock, Shield, ShieldCheck, UserCog } from "lucide-react";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { useStore, type DashboardRole } from "@/lib/store";

const stagger = {
  hidden: { opacity: 0 },
  show: { opacity: 1, transition: { staggerChildren: 0.08 } },
};
const fadeUp = {
  hidden: { opacity: 0, y: 12 },
  show: { opacity: 1, y: 0, transition: { duration: 0.4, ease: "easeOut" as const } },
} as const;

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
  } = useStore(useShallow((state) => ({
      user: state.user,
      role: state.role,
      mfaStatus: state.mfaStatus,
      mfaEnrollment: state.mfaEnrollment,
      adminUsers: state.adminUsers,
      securityStatus: state.securityStatus,
      ssoStatus: state.ssoStatus,
      siemStatus: state.siemStatus,
      hasPermission: state.hasPermission,
      fetchSsoStatus: state.fetchSsoStatus,
      fetchAdminUsers: state.fetchAdminUsers,
      fetchSecurityStatus: state.fetchSecurityStatus,
      fetchSiemStatus: state.fetchSiemStatus,
      initMfaEnrollment: state.initMfaEnrollment,
      verifyMfaEnrollment: state.verifyMfaEnrollment,
      disableMfaEnrollment: state.disableMfaEnrollment,
      updateUserRole: state.updateUserRole,
      rotateJwtSecret: state.rotateJwtSecret,
      resetKillSwitch: state.resetKillSwitch,
      flushSiemQueue: state.flushSiemQueue,
    })));

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
    if (canManageUsers) void fetchAdminUsers();
    if (canRotateSecurity) void fetchSecurityStatus();
    if (canReadAudit) void fetchSiemStatus();
  }, [canManageUsers, canRotateSecurity, canReadAudit, fetchSsoStatus, fetchAdminUsers, fetchSecurityStatus, fetchSiemStatus]);

  const effectiveTargetUser = targetUser || adminUsers[0]?.username || "";
  const selectedAdminUser = useMemo(
    () => adminUsers.find((item) => item.username === effectiveTargetUser),
    [adminUsers, effectiveTargetUser],
  );

  return (
    <motion.div variants={stagger} initial="hidden" animate="show" className="space-y-6">
      {/* Header */}
      <motion.section variants={fadeUp} className="rounded-2xl border border-white/[0.08] bg-white/[0.03] p-6 backdrop-blur-sm">
        <p className="text-[10px] font-semibold uppercase tracking-[0.2em] text-cyan-500/70">Control Settings</p>
        <h1 className="mt-2 text-3xl font-bold text-slate-100">Security & Governance</h1>
        <p className="mt-1 text-sm text-slate-400">Role access, MFA enrollment, key rotation and privileged controls.</p>
        <div className="mt-3 flex flex-wrap gap-2">
          <Badge variant="outline">Role {role.toUpperCase()}</Badge>
          <Badge variant={mfaStatus?.mfa_enabled ? "success" : "warning"}>
            MFA {mfaStatus?.mfa_enabled ? "ENABLED" : "DISABLED"}
          </Badge>
        </div>
      </motion.section>

      {/* Kill Switch Reset + MFA */}
      <motion.div variants={fadeUp} className="grid gap-4 lg:grid-cols-2">
        <Card>
          <CardHeader>
            <CardTitle className="flex items-center gap-2">
              <Shield size={18} className="text-rose-400" />
              Kill Switch Reset Controls
            </CardTitle>
            <CardDescription>Publishes a reset control event to the backend control bus.</CardDescription>
          </CardHeader>
          <CardContent className="space-y-3">
            <label className="block text-sm">
              <span className="mb-1 block text-xs uppercase tracking-wider text-slate-500">Authorized By</span>
              <input className="glass-input h-10 w-full" value={authorizedBy} onChange={(e) => setAuthorizedBy(e.target.value)} />
            </label>
            <label className="flex items-center gap-2 text-sm text-slate-300">
              <input type="checkbox" checked={force} onChange={(e) => setForce(e.target.checked)} className="accent-cyan-500" />
              Force reset (requires override code)
            </label>
            <label className="block text-sm">
              <span className="mb-1 block text-xs uppercase tracking-wider text-slate-500">Override Code</span>
              <input className="glass-input h-10 w-full" value={overrideCode} onChange={(e) => setOverrideCode(e.target.value)} placeholder="Required only for force reset" />
            </label>
            {requiresMfa && (
              <label className="block text-sm">
                <span className="mb-1 block text-xs uppercase tracking-wider text-slate-500">MFA Code</span>
                <input className="glass-input h-10 w-full font-mono" value={mfaCode} onChange={(e) => setMfaCode(e.target.value.replace(/\D/g, "").slice(0, 6))} placeholder="6-digit TOTP" />
              </label>
            )}
            <Button
              variant="destructive"
              onClick={() => void resetKillSwitch(authorizedBy, force, overrideCode || undefined, mfaCode || undefined)}
              disabled={!canResetKillSwitch || (requiresMfa && mfaCode.length !== 6)}
            >
              Publish Reset Command
            </Button>
            {!canResetKillSwitch && <p className="text-xs text-rose-400">Role lacks kill-switch reset permission.</p>}
          </CardContent>
        </Card>

        <Card>
          <CardHeader>
            <CardTitle className="flex items-center gap-2">
              <Lock size={18} className="text-cyan-400" />
              MFA Enrollment
            </CardTitle>
            <CardDescription>Enroll with TOTP and enforce MFA on privileged actions.</CardDescription>
          </CardHeader>
          <CardContent className="space-y-3">
            {!mfaStatus?.mfa_enabled ? (
              <>
                <Button variant="outline" onClick={() => void initMfaEnrollment()}>
                  Generate Enrollment Secret
                </Button>
                {mfaEnrollment && (
                  <div className="rounded-lg border border-white/[0.08] bg-white/[0.03] p-3 text-xs">
                    <p className="font-semibold text-slate-300">Secret: <span className="font-mono text-cyan-400">{mfaEnrollment.secret}</span></p>
                    <p className="mt-1 break-all text-slate-500">{mfaEnrollment.provisioning_uri}</p>
                    <label className="mt-3 block text-sm">
                      <span className="mb-1 block text-xs uppercase tracking-wider text-slate-500">Verification Code</span>
                      <input className="glass-input h-10 w-full font-mono" value={enrollCode} onChange={(e) => setEnrollCode(e.target.value.replace(/\D/g, "").slice(0, 6))} placeholder="6-digit code" />
                    </label>
                    <Button className="mt-2" onClick={() => void verifyMfaEnrollment(enrollCode)} disabled={enrollCode.length !== 6}>
                      Verify & Enable MFA
                    </Button>
                  </div>
                )}
              </>
            ) : (
              <>
                <p className="text-sm text-emerald-400">MFA is enabled for this account.</p>
                <label className="block text-sm">
                  <span className="mb-1 block text-xs uppercase tracking-wider text-slate-500">Disable MFA Code</span>
                  <input className="glass-input h-10 w-full font-mono" value={disableCode} onChange={(e) => setDisableCode(e.target.value.replace(/\D/g, "").slice(0, 6))} placeholder="6-digit code" />
                </label>
                <Button variant="outline" onClick={() => void disableMfaEnrollment(disableCode)} disabled={disableCode.length !== 6}>
                  Disable MFA
                </Button>
              </>
            )}
          </CardContent>
        </Card>
      </motion.div>

      {/* User Management + JWT */}
      <motion.div variants={fadeUp} className="grid gap-4 lg:grid-cols-2">
        {canManageUsers && (
          <Card>
            <CardHeader>
              <CardTitle className="flex items-center gap-2">
                <UserCog size={18} className="text-amber-400" />
                Admin User Role Management
              </CardTitle>
              <CardDescription>Manage runtime role overrides for dashboard users.</CardDescription>
            </CardHeader>
            <CardContent className="space-y-3">
              <label className="block text-sm">
                <span className="mb-1 block text-xs uppercase tracking-wider text-slate-500">User</span>
                <select
                  className="glass-select h-10 w-full"
                  value={effectiveTargetUser}
                  onChange={(e) => {
                    const nextUser = e.target.value;
                    setTargetUser(nextUser);
                    const candidate = adminUsers.find((item) => item.username === nextUser)?.role as DashboardRole | undefined;
                    if (candidate && roleOptions.includes(candidate)) {
                      setTargetRole(candidate);
                    }
                  }}
                >
                  {adminUsers.map((item) => (
                    <option key={item.username} value={item.username}>{item.username}</option>
                  ))}
                </select>
              </label>
              <label className="block text-sm">
                <span className="mb-1 block text-xs uppercase tracking-wider text-slate-500">Role</span>
                <select className="glass-select h-10 w-full" value={targetRole} onChange={(e) => setTargetRole(e.target.value as DashboardRole)}>
                  {roleOptions.map((item) => (
                    <option key={item} value={item}>{item}</option>
                  ))}
                </select>
              </label>
              {requiresMfa && (
                <label className="block text-sm">
                  <span className="mb-1 block text-xs uppercase tracking-wider text-slate-500">Admin MFA Code</span>
                  <input className="glass-input h-10 w-full font-mono" value={mfaCode} onChange={(e) => setMfaCode(e.target.value.replace(/\D/g, "").slice(0, 6))} placeholder="6-digit code" />
                </label>
              )}
              <Button variant="outline" onClick={() => void updateUserRole(effectiveTargetUser, targetRole, "dashboard_admin_update", mfaCode || undefined)} disabled={!effectiveTargetUser || (requiresMfa && mfaCode.length !== 6)}>
                Update Role
              </Button>
              {selectedAdminUser && (
                <div className="rounded-lg border border-white/[0.08] bg-white/[0.03] p-3 text-xs text-slate-400">
                  <p>Current role: <span className="font-semibold text-slate-200">{selectedAdminUser.role}</span></p>
                  <p>Role source: {selectedAdminUser.role_source}</p>
                  <p>MFA enabled: {selectedAdminUser.has_mfa ? "yes" : "no"}</p>
                </div>
              )}
            </CardContent>
          </Card>
        )}

        {canRotateSecurity && (
          <Card>
            <CardHeader>
              <CardTitle className="flex items-center gap-2">
                <Key size={18} className="text-cyan-400" />
                JWT Key Rotation
              </CardTitle>
              <CardDescription>Rotate signing key with short key-ring fallback for token continuity.</CardDescription>
            </CardHeader>
            <CardContent className="space-y-3">
              <div className="rounded-lg border border-white/[0.08] bg-white/[0.03] p-3 text-xs text-slate-400">
                <p>Active key fingerprint: <span className="font-mono text-slate-200">{securityStatus?.active_key_fingerprint ?? "--"}</span></p>
                <p>Key ring size: {securityStatus?.jwt_key_count ?? 0}</p>
                <p>Rate limit: {securityStatus?.rate_limit_enabled ? "enabled" : "disabled"} ({securityStatus?.rate_limit_limit ?? 0}/{securityStatus?.rate_limit_window_seconds ?? 0}s)</p>
              </div>
              <label className="block text-sm">
                <span className="mb-1 block text-xs uppercase tracking-wider text-slate-500">New JWT Secret</span>
                <input className="glass-input h-10 w-full font-mono" value={rotateSecret} onChange={(e) => setRotateSecret(e.target.value)} placeholder="at least 32 chars" />
              </label>
              {requiresMfa && (
                <label className="block text-sm">
                  <span className="mb-1 block text-xs uppercase tracking-wider text-slate-500">MFA Code</span>
                  <input className="glass-input h-10 w-full font-mono" value={mfaCode} onChange={(e) => setMfaCode(e.target.value.replace(/\D/g, "").slice(0, 6))} placeholder="6-digit code" />
                </label>
              )}
              <Button variant="outline" onClick={() => void rotateJwtSecret(rotateSecret, mfaCode || undefined)} disabled={rotateSecret.length < 32 || (requiresMfa && mfaCode.length !== 6)}>
                Rotate JWT Secret
              </Button>
            </CardContent>
          </Card>
        )}
      </motion.div>

      {/* SSO + SIEM */}
      <motion.div variants={fadeUp} className="grid gap-4 lg:grid-cols-2">
        <Card>
          <CardHeader>
            <CardTitle className="flex items-center gap-2">
              <Globe size={18} className="text-cyan-400" />
              Enterprise SSO
            </CardTitle>
            <CardDescription>OIDC federation status for institutional identity providers.</CardDescription>
          </CardHeader>
          <CardContent className="space-y-2 text-sm text-slate-300">
            <p>Enabled: <span className="font-semibold">{ssoStatus?.enabled ? "yes" : "no"}</span></p>
            <p>Configured: <span className="font-semibold">{ssoStatus?.configured ? "yes" : "no"}</span></p>
            <p>Issuer: <span className="font-mono text-xs text-slate-400">{ssoStatus?.issuer ?? "--"}</span></p>
            <p>Role claim: <span className="font-mono text-xs text-slate-400">{ssoStatus?.role_claim ?? "--"}</span></p>
          </CardContent>
        </Card>

        {canReadAudit && (
          <Card>
            <CardHeader>
              <CardTitle className="flex items-center gap-2">
                <ShieldCheck size={18} className="text-emerald-400" />
                SIEM Delivery
              </CardTitle>
              <CardDescription>Forwarding status for centralized audit pipeline.</CardDescription>
            </CardHeader>
            <CardContent className="space-y-2 text-sm text-slate-300">
              <p>Enabled: <span className="font-semibold">{siemStatus?.enabled ? "yes" : "no"}</span></p>
              <p>Endpoint: <span className="font-mono text-xs text-slate-400">{siemStatus?.endpoint ?? "--"}</span></p>
              <p>Queue depth: <span className="font-semibold">{siemStatus?.queue_depth ?? 0}</span></p>
              {siemStatus?.last_error && <p className="text-xs text-rose-400">{siemStatus.last_error}</p>}
              <Button variant="outline" size="sm" onClick={() => void flushSiemQueue(5)} disabled={!canManageAudit}>
                Flush SIEM Queue
              </Button>
            </CardContent>
          </Card>
        )}
      </motion.div>
    </motion.div>
  );
}


