# Naming

This public package uses the final paper-facing names consistently.

| Name | Meaning |
| --- | --- |
| `G` | Frozen DAv2 high-level feature branch injected into the fnet matching feature stream. |
| `Z` | Frozen DAv2 depth map injected into the cnet/context input stream. |
| `DAB-Smooth` | Depth-aware boundary smoothness regularizer. |
| `Depth-RAFT G+Z+DAB` | Final method. |

Internal checkpoint keys still include `dav2_fusion` because that is the serialized module name. It is an implementation key, not a public method name.
