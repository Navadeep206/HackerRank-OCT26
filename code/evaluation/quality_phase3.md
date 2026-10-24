# Code quality improvement — Phase 3: decision quality and safety

## Changes

- Corrected primary issue/part association for conversational corrections. When a user says that one part is fine but a different part is damaged, the parser associates the primary damage with the part nearest the positive damage mention instead of blindly selecting the first part mentioned.
- Added a regression assertion for a claim such as "the screen is fine, actually the keyboard is cracked, not the screen."
- Added decision tests proving that history risk flags cannot flip a visually contradicted claim.
- Added a decision test that verifies the confidence threshold is applied as configured.
- Retained conservative behavior for low-confidence findings, unclear targets, conflicting clear images, and prompt-injection text.

## Safety contract

Claim parsing extracts intent; it does not establish proof. Validated image findings and the deterministic decision policy determine claim status. User-history flags add review context only and must never independently support or contradict damage. Prompt-injection text is treated as untrusted claim content and flagged, not executed.

## Validation scope

The added tests use deterministic findings and local images. They do not call a live vision model. Passing these tests verifies the covered policy cases, not overall real-world claim accuracy.
