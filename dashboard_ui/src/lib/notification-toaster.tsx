import { Toaster } from "sonner";

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
