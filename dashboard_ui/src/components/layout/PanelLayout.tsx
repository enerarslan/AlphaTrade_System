import { type ReactNode } from "react";
import { Panel, Group, Separator, useDefaultLayout } from "react-resizable-panels";

interface PanelConfig {
  id: string;
  defaultSize?: number;
  minSize?: number;
  children: ReactNode;
  collapsible?: boolean;
}

interface PanelLayoutProps {
  /** Layout orientation */
  orientation?: "horizontal" | "vertical";
  /** Panels to render */
  panels: PanelConfig[];
  /** Storage key for layout persistence */
  storageKey?: string;
  /** Additional className */
  className?: string;
}

function ResizeHandle({ orientation }: { orientation: "horizontal" | "vertical" }) {
  return (
    <Separator className="group relative flex items-center justify-center">
      {orientation === "horizontal" ? (
        <div className="mx-0.5 h-full w-[3px] rounded-full bg-white/[0.06] transition-all group-hover:bg-cyan-500/30 group-data-[resize-handle-active]:bg-cyan-500/60 group-data-[resize-handle-active]:shadow-[0_0_6px_rgba(6,182,212,0.3)]" />
      ) : (
        <div className="my-0.5 h-[3px] w-full rounded-full bg-white/[0.06] transition-all group-hover:bg-cyan-500/30 group-data-[resize-handle-active]:bg-cyan-500/60 group-data-[resize-handle-active]:shadow-[0_0_6px_rgba(6,182,212,0.3)]" />
      )}
    </Separator>
  );
}

/** A persistent panel layout wrapper */
function PersistentPanelLayout({
  orientation = "horizontal",
  panels,
  storageKey,
  className = "",
}: PanelLayoutProps & { storageKey: string }) {
  const { defaultLayout, onLayoutChanged } = useDefaultLayout({
    id: storageKey,
    panelIds: panels.map((p) => p.id),
  });

  return (
    <Group
      orientation={orientation}
      defaultLayout={defaultLayout}
      onLayoutChanged={onLayoutChanged}
      className={`rounded-xl ${className}`}
    >
      {panels.map((panel, i) => (
        <PanelItem key={panel.id} panel={panel} isLast={i === panels.length - 1} orientation={orientation} />
      ))}
    </Group>
  );
}

/** A non-persistent panel layout */
function SimplePanelLayout({
  orientation = "horizontal",
  panels,
  className = "",
}: Omit<PanelLayoutProps, "storageKey">) {
  return (
    <Group
      orientation={orientation}
      className={`rounded-xl ${className}`}
    >
      {panels.map((panel, i) => (
        <PanelItem key={panel.id} panel={panel} isLast={i === panels.length - 1} orientation={orientation} />
      ))}
    </Group>
  );
}

export default function PanelLayout({
  orientation = "horizontal",
  panels,
  storageKey,
  className = "",
}: PanelLayoutProps) {
  if (storageKey) {
    return (
      <PersistentPanelLayout
        orientation={orientation}
        panels={panels}
        storageKey={storageKey}
        className={className}
      />
    );
  }
  return (
    <SimplePanelLayout
      orientation={orientation}
      panels={panels}
      className={className}
    />
  );
}

function PanelItem({
  panel,
  isLast,
  orientation,
}: {
  panel: PanelConfig;
  isLast: boolean;
  orientation: "horizontal" | "vertical";
}) {
  return (
    <>
      <Panel
        id={panel.id}
        defaultSize={panel.defaultSize != null ? `${panel.defaultSize}%` : undefined}
        minSize={panel.minSize != null ? `${panel.minSize}%` : "15%"}
        collapsible={panel.collapsible}
        className="overflow-auto"
      >
        {panel.children}
      </Panel>
      {!isLast && <ResizeHandle orientation={orientation} />}
    </>
  );
}

/** Preset layouts for common page arrangements */
export const PANEL_PRESETS = {
  /** Two columns: 60/40 split */
  twoColumn: (left: ReactNode, right: ReactNode, storageKey?: string) => (
    <PanelLayout
      orientation="horizontal"
      storageKey={storageKey}
      panels={[
        { id: "left", defaultSize: 60, minSize: 30, children: left },
        { id: "right", defaultSize: 40, minSize: 25, children: right },
      ]}
    />
  ),

  /** Stacked: top 60% / bottom 40% */
  stacked: (top: ReactNode, bottom: ReactNode, storageKey?: string) => (
    <PanelLayout
      orientation="vertical"
      storageKey={storageKey}
      panels={[
        { id: "top", defaultSize: 60, minSize: 25, children: top },
        { id: "bottom", defaultSize: 40, minSize: 20, children: bottom },
      ]}
    />
  ),
};
