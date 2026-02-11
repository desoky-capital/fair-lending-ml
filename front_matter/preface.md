# Preface

---

## Why I Wrote This Book

In 2019, headlines about the Apple Card algorithm allegedly offering women lower credit limits than men caught my attention—not just as a news story, but as a wake-up call. Here was one of the world's most sophisticated technology companies, partnering with one of the world's largest banks, deploying an algorithm that appeared to discriminate by gender. The algorithm never explicitly considered gender. It didn't need to. It found proxies.

This wasn't a failure of intent. It was a failure of process.

As someone who has spent years at the intersection of finance and technology—first as an investment banker advising on complex transactions, now as an academic director preparing the next generation of fintech leaders—I recognized a troubling gap. The students I teach are eager to build machine learning models. The practitioners I meet are deploying them at scale. But too few recognize the unique challenges of building ML systems in regulated financial environments where algorithmic decisions affect people's access to credit, housing, and economic opportunity.

There's no shortage of resources on machine learning. There's growing literature on AI ethics. There are regulatory guidelines and academic papers on fairness metrics. But there's a gap: **practical, accessible guidance for anyone who needs to evaluate, build, or oversee fair ML systems in financial services.**

This book exists to fill that gap.

---

## What Makes This Book Different

**First, it's honest about failure.**

Most ML tutorials show you how to build a model that works. This book shows you how models fail—and why that matters more. In Chapter 3, we build a credit model that achieves 95% accuracy on validation data, then watch it collapse completely on test data. We don't hide this failure; we learn from it. Because in the real world, knowing why models fail is more valuable than celebrating when they succeed.

**Second, it's accessible to non-coders.**

You don't need to write a single line of Python to grasp algorithmic fairness. Every technical concept is explained in plain language. Every code block is preceded by explanation and followed by interpretation. Whether you're a data scientist building models, a compliance officer validating them, or an executive overseeing AI strategy, this book meets you where you are.

**Third, it's grounded in regulatory reality.**

Credit models aren't just technical artifacts—they're regulated instruments with real legal consequences. This book integrates regulatory requirements (ECOA, FCRA, SR 11-7, the EU AI Act) throughout, not as an afterthought but as a fundamental design constraint. Because in financial services, a model that's accurate but discriminatory isn't just unethical—it's illegal.

**Fourth, it bridges theory and practice.**

We don't just explain fairness metrics; we implement them. We don't just discuss bias mitigation; we compare approaches and show you which ones work (and which ones destroy your model). We don't just mention monitoring; we build a complete system. By the end, you'll have both conceptual knowledge and practical tools.

---

## Who This Book Is For

**Data scientists and ML engineers** who build models that affect people's lives and want to do it responsibly.

**Compliance and risk professionals** who need to know what's technically possible and what questions to ask.

**Managers and executives** who oversee AI strategy and need informed perspective on fairness trade-offs.

**Students and career changers** preparing for roles in fintech, AI ethics, or responsible ML.

**Anyone curious** about how algorithms can discriminate—and how we can build them better.

---

## A Personal Note

My path to this book has been unconventional. West Point taught me about leadership and accountability. Investment banking taught me about high-stakes decision-making under uncertainty. Academia has taught me the joy of helping others learn.

But perhaps most relevant is what the Army taught me: that systems matter. Good people in bad systems produce bad outcomes. Bad people in good systems get caught. The same is true for algorithms. A well-intentioned team with poor processes will ship discriminatory models. A rigorous team with good processes will catch problems before they cause harm.

This book is about building good systems—technical systems, yes, but also organizational systems, documentation systems, and monitoring systems. Because fairness isn't a one-time check; it's an ongoing commitment.

---

## Acknowledgments

This book would not have been possible without the support of many people.

To my students at Wake Forest University's School of Professional Studies: your questions, challenges, and enthusiasm have shaped every page. Teaching you has taught me.

To my colleagues in academia and industry who reviewed drafts, tested code, and offered feedback: your insights made this book immeasurably better.

To the researchers whose work on algorithmic fairness laid the foundation for practitioners like me: thank you for asking the hard questions first.

To my wife, Erica: thank you for your patience during late nights of writing and your unwavering support for yet another ambitious project. Life near the water with you makes everything better.

And to every practitioner who picks up this book with the genuine desire to build fairer systems: thank you for caring enough to try.

---

## How to Use This Book

This book offers two reading paths:

**Path A (Hands-On):** Run every code block, experiment with variations, and build the complete model yourself. Time: 15-20 hours.

**Path B (Conceptual):** Read the explanations and results, skip the code, and focus on grasping concepts and trade-offs. Time: 6-8 hours.

Both paths lead to genuine comprehension. Choose based on your goals and background.

For those using this book in a classroom or study group, each chapter ends with discussion questions, exercises, and teaching notes.

---

## Let's Begin

The stakes are high. Algorithms are making decisions that affect people's access to credit, housing, insurance, and opportunity. Those algorithms can perpetuate historical discrimination or help overcome it. The choice depends on how we build them.

You're holding a guide to building them right.

**The tools are here. The techniques are proven. The need is urgent.**

Let's build something fair.

---

*Mohamed Desoky, PhD, MBA*  
*Florida, 2026*
