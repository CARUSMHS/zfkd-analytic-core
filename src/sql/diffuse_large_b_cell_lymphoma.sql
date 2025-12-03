DROP TABLE IF EXISTS cdm.analytics_diffuse_large_b_cell_lymphoma;
CREATE TABLE cdm.analytics_diffuse_large_b_cell_lymphoma AS

-- persons diagnosed with diffuse large B-cell lymphoma
-- consider each person only once (only first diagnosis with tumor)
WITH lymphoma AS (
    SELECT
        person_id,
        provider_id,
        condition_source_concept_id,
        condition_start_date AS diagnosis_date
    FROM (
    SELECT
        person_id,
        provider_id,
        condition_start_date,
        condition_concept_id,
        condition_source_concept_id,
        ROW_NUMBER() OVER (
        PARTITION BY person_id, condition_concept_id
        ORDER BY condition_start_date
        ) AS rn
        FROM cdm.condition_occurrence
        WHERE condition_concept_id = 432574
    ) as first_diagnosis
    WHERE rn = 1
),

person_info AS (
    SELECT
        person_id,
        gender_concept_id,
        year_of_birth
    FROM
        cdm.person
),

observation_end AS (
    SELECT
        person_id,
        observation_period_end_date AS observation_end_date
    FROM
        cdm.observation_period
),

-- consider each person only once (persons can be duble in death table as they may have multiple death reasons)
death_info AS (
    SELECT
        person_id,
        1 AS death_flag
    FROM (
        SELECT
            person_id,
            ROW_NUMBER() OVER (PARTITION BY person_id ORDER BY death_date) AS rn
        FROM cdm.death
    ) sub
    WHERE rn = 1
)

-- Final query
SELECT
    lymphoma.person_id,
    pi.year_of_birth,
    pi.gender_concept_id,
    lymphoma.provider_id,
    lymphoma.condition_source_concept_id,
    lymphoma.diagnosis_date,
    oe.observation_end_date,
    COALESCE(di.death_flag, 0) AS death_flag
FROM
    lymphoma
LEFT JOIN person_info AS pi
    ON lymphoma.person_id = pi.person_id
LEFT JOIN observation_end AS oe
    ON lymphoma.person_id = oe.person_id
LEFT JOIN death_info AS di
    ON lymphoma.person_id = di.person_id;
