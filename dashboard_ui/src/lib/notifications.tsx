import { Toaster, toast } from "sonner";

/** Mount this once at the root of the app (inside DashboardLayout). */
export function NotificationToaster() {
  return (
    <Toaster
      position="top-right"
      theme="dark"
      richColors
      closeButton
      offset={16}
      gap={8}
    />
  );
}

type Severity = "CRITICAL" | "HIGH" | "MEDIUM" | "LOW" | "INFO";

/** Push a trading notification. Called from WS alert handler. */
export function pushNotification(
  title: string,
  message: string,
  severity: Severity = "INFO",
) {
  switch (severity) {
    case "CRITICAL":
      toast.error(title, {
        description: message,
        duration: Infinity,
      });
      break;
    case "HIGH":
      toast.warning(title, {
        description: message,
        duration: 10000,
      });
      break;
    case "MEDIUM":
      toast.info(title, {
        description: message,
        duration: 6000,
      });
      break;
    default:
      toast(title, {
        description: message,
        duration: 4000,
      });
  }
}
