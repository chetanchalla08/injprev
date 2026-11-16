"""Evidence registry for health-related interpretations.

A finding may carry a health-related interpretation ONLY by pointing at an entry here.
Bibliographic details were checked against public records when the entries were written,
but the summaries and their wording have NOT been reviewed by a sports clinician
(review_status below). That review is a release requirement; see docs/VALIDATION.md.
"""
from __future__ import annotations

REVIEW_STATUS = "Not yet reviewed by a sports clinician"

EVIDENCE: dict[str, dict] = {
    "elliott2003": {
        "citation": "Elliott B, Fleisig G, Nicholls R, Escamilla R. Technique effects on upper limb loading in the tennis serve. J Sci Med Sport. 2003;6(1):76-87.",
        "doi": "10.1016/S1440-2440(03)80011-7",
        "population": "High-level players serving in competition, measured with multi-camera 3D analysis.",
        "summary": "As a group, servers who bent their knees more in the backswing showed lower peak shoulder internal-rotation torque than servers who bent less.",
        "limits": "Group-level association in skilled players; not a threshold, not a prediction for an individual, and not shown to transfer to recreational players. Joint torque cannot be measured from ordinary video.",
    },
    "campbell2013": {
        "citation": "Campbell A, Straker L, O'Sullivan P, Elliott B, Reid M. Lumbar loading in the elite adolescent tennis serve: link to low back pain. Med Sci Sports Exerc. 2013;45(8):1562-1568.",
        "doi": "10.1249/MSS.0b013e31828bea5e",
        "population": "20 elite adolescent male players (7 with a history of disabling low back pain), laboratory 3D motion capture.",
        "summary": "Sideways (lateral) forces on the lower back were high in all servers, and the players with a back-pain history showed somewhat higher peak lateral force than matched controls.",
        "limits": "Small, cross-sectional study of elite adolescents; it cannot show cause. Trunk tilt seen in a 2D video is not a force measurement, and substantial sideways lean is a normal part of serving.",
    },
    "hewett2005": {
        "citation": "Hewett TE, Myer GD, Ford KR, et al. Biomechanical measures of neuromuscular control and valgus loading of the knee predict anterior cruciate ligament injury risk in female athletes: a prospective study. Am J Sports Med. 2005;33(4):492-501.",
        "doi": "10.1177/0363546504269591",
        "population": "205 adolescent female soccer, basketball and volleyball players; laboratory 3D jump-landing test.",
        "summary": "Athletes who later had an ACL injury had shown more inward knee motion and loading during a jump-landing test.",
        "limits": "Not tennis, not 2D video, and a later, larger study did not reproduce the result (see Krosshaug 2016).",
    },
    "krosshaug2016": {
        "citation": "Krosshaug T, Steffen K, Kristianslund E, et al. The vertical drop jump is a poor screening test for ACL injuries in female elite soccer and handball players: a prospective cohort study of 710 athletes. Am J Sports Med. 2016;44(4):874-883.",
        "doi": "10.1177/0363546515625048",
        "population": "710 elite female soccer and handball players followed prospectively.",
        "summary": "Inward knee displacement during a jump-landing test did not usefully identify who went on to an ACL injury.",
        "limits": "Not tennis. Included to show that the evidence on inward knee motion as a screening sign is mixed.",
    },
    "gabbett2016": {
        "citation": "Gabbett TJ. The training-injury prevention paradox: should athletes be training smarter and harder? Br J Sports Med. 2016;50(5):273-280.",
        "doi": "10.1136/bjsports-2015-095788",
        "population": "Narrative review, mostly team sports.",
        "summary": "Proposes that rapid increases in training load relative to what an athlete is used to are associated with more injuries, while well-built-up high loads may be protective.",
        "limits": "Mostly team-sport data; the workload-ratio approach has been strongly criticised (see Impellizzeri 2020).",
    },
    "impellizzeri2020": {
        "citation": "Impellizzeri FM, Tenan MS, Kempton T, Novak A, Coutts AJ. Acute:chronic workload ratio: conceptual issues and fundamental pitfalls. Int J Sports Physiol Perform. 2020;15(6):907-913.",
        "doi": "10.1123/ijspp.2019-0864",
        "population": "Methodological critique.",
        "summary": "Argues there is no sound evidence for using workload ratios to make training recommendations aimed at reducing injury.",
        "limits": "A critique of one metric; it does not show that sudden workload changes are harmless.",
    },
}


def refs(*ids: str) -> list[dict]:
    return [{"id": i, **EVIDENCE[i], "review_status": REVIEW_STATUS} for i in ids]
