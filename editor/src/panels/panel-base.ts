/**
 * Panel base class.
 *
 * Each of the 8 editor panels is a lightweight vanilla-DOM component.
 * Panels receive an `EditorState` reference at construction and subscribe
 * to its event stream. Panels render themselves into a provided container.
 *
 * No framework dependency — panels are pure TypeScript + DOM API.
 */

import { EditorState, EditorEvent } from "../editor-state.js";

export abstract class Panel {
  protected root: HTMLElement;
  protected state: EditorState;
  private _unsub: () => void = () => {};

  constructor(container: HTMLElement, state: EditorState) {
    this.root = container;
    this.state = state;
    this._unsub = state.on((ev: EditorEvent) => this.onEditorEvent(ev));
    this.render();
  }

  /** Respond to editor state changes. Subclasses override selectively. */
  protected onEditorEvent(_event: EditorEvent): void {}

  /** Full re-render of the panel's content. Called once at construction. */
  protected abstract render(): void;

  /** Clean up subscriptions (call when removing a panel from the DOM). */
  destroy(): void {
    this._unsub();
  }

  /** Convenience: clear and re-render. */
  protected refresh(): void {
    this.root.innerHTML = "";
    this.render();
  }

  protected el<K extends keyof HTMLElementTagNameMap>(
    tag: K,
    attrs: Partial<HTMLElementTagNameMap[K]> & { class?: string; text?: string } = {}
  ): HTMLElementTagNameMap[K] {
    const el = document.createElement(tag);
    if (attrs.class) el.className = attrs.class;
    if (attrs.text) el.textContent = attrs.text;
    return el;
  }
}
