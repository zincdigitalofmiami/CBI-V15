/** @type {import('dependency-cruiser').IConfiguration} */
module.exports = {
  forbidden: [
    /* RULES FOR ISOLATION */
    {
      name: "legislation-isolation",
      severity: "error",
      comment:
        "Isolation: Files in app/(pages)/legislation cannot import from app/(pages)/strategy.",
      from: {
        path: "^app/\\(pages\\)/legislation",
      },
      to: {
        path: "^app/\\(pages\\)/strategy",
      },
    },
    {
      name: "strategy-isolation",
      severity: "error",
      comment:
        "Isolation: Files in app/(pages)/strategy cannot import from app/(pages)/legislation.",
      from: {
        path: "^app/\\(pages\\)/strategy",
      },
      to: {
        path: "^app/\\(pages\\)/legislation",
      },
    },
    /* RULES FOR CONFIG IMPORTS */
    /* Note: Enforcing '@/' alias usage vs relative paths for the same file is better handled by ESLint.
       dependency-cruiser validates architectural dependencies (A depends on B), not the import string syntax used.
    */

    /* RULES FOR CIRCULAR DEPENDENCIES */
    {
      name: "no-circular",
      severity: "error",
      comment: "No Circulars: Ban all circular dependencies.",
      from: {},
      to: {
        circular: true,
      },
    },
  ],
  options: {
    /* options to make the tool easier to work with */
    doNotFollow: {
      path: "node_modules",
    },
    tsPreCompilationDeps: true, // helpful for TypeScript
    tsConfig: {
      fileName: "tsconfig.json",
    },
  },
};
