package main

import (
	"fmt"

	"gopkg.in/yaml.v3"
)

// validateNode dispatches to per-file validation based on name.
func validateNode(name string, root *yaml.Node) []string {
	switch name {
	case "certifications":
		return validateCertifications(root)
	case "skills":
		return validateSkills(root)
	case "projects":
		return validateProjects(root)
	case "experience":
		return validateExperience(root)
	}
	return nil
}

// --- certifications.yaml ---
// Top-level key: certifications (list)
// Each item: id, name, issuer (string), ats_keywords (string list), relevance (string list)

func validateCertifications(root *yaml.Node) []string {
	var errs []string
	top := nodeMapping(root)
	errs = append(errs, requireKeys(top, "certifications.yaml", []string{"certifications"})...)
	certsNode, ok := top["certifications"]
	if !ok {
		return errs
	}
	if certsNode.Kind != yaml.SequenceNode {
		errs = append(errs, "certifications: expected list")
		return errs
	}
	for i, item := range certsNode.Content {
		path := fmt.Sprintf("certifications[%d]", i)
		if item.Kind != yaml.MappingNode {
			errs = append(errs, fmt.Sprintf("%s: expected map", path))
			continue
		}
		m := nodeMapping(item)
		errs = append(errs, requireKeys(m, path, []string{"id", "name", "issuer", "ats_keywords", "relevance"})...)
		if n, ok := m["ats_keywords"]; ok {
			errs = append(errs, nodeStringList(n, path+".ats_keywords")...)
		}
		if n, ok := m["relevance"]; ok {
			errs = append(errs, nodeStringList(n, path+".relevance")...)
		}
	}
	return errs
}

// --- skills.yaml ---
// Top-level key: categories (list)
// Each category: name (string), skills (list)
// Each skill: name (string), ats_keywords (string list), proficiency (enum), evidence (list)
// Each evidence: type (enum "role"|"project"|"cert"), ref (string), detail (optional string)

func validateSkills(root *yaml.Node) []string {
	var errs []string
	top := nodeMapping(root)
	errs = append(errs, requireKeys(top, "skills.yaml", []string{"categories"})...)
	catsNode, ok := top["categories"]
	if !ok {
		return errs
	}
	if catsNode.Kind != yaml.SequenceNode {
		errs = append(errs, "categories: expected list")
		return errs
	}
	for i, cat := range catsNode.Content {
		catPath := fmt.Sprintf("categories[%d]", i)
		if cat.Kind != yaml.MappingNode {
			errs = append(errs, fmt.Sprintf("%s: expected map", catPath))
			continue
		}
		cm := nodeMapping(cat)
		errs = append(errs, requireKeys(cm, catPath, []string{"name", "skills"})...)
		skillsNode, ok := cm["skills"]
		if !ok {
			continue
		}
		if skillsNode.Kind != yaml.SequenceNode {
			errs = append(errs, fmt.Sprintf("%s.skills: expected list", catPath))
			continue
		}
		for j, skill := range skillsNode.Content {
			skillPath := fmt.Sprintf("%s.skills[%d]", catPath, j)
			if skill.Kind != yaml.MappingNode {
				errs = append(errs, fmt.Sprintf("%s: expected map", skillPath))
				continue
			}
			sm := nodeMapping(skill)
			errs = append(errs, requireKeys(sm, skillPath, []string{"name", "ats_keywords", "proficiency"})...)
			if n, ok := sm["ats_keywords"]; ok {
				errs = append(errs, nodeStringList(n, skillPath+".ats_keywords")...)
			}
			if n, ok := sm["proficiency"]; ok {
				errs = append(errs, enumCheck(n, skillPath+".proficiency", []string{"advanced", "intermediate", "familiar"})...)
			}
			if evidNode, ok := sm["evidence"]; ok {
				if evidNode.Kind != yaml.SequenceNode {
					errs = append(errs, fmt.Sprintf("%s.evidence: expected list", skillPath))
				} else {
					for k, ev := range evidNode.Content {
						evPath := fmt.Sprintf("%s.evidence[%d]", skillPath, k)
						if ev.Kind != yaml.MappingNode {
							errs = append(errs, fmt.Sprintf("%s: expected map", evPath))
							continue
						}
						evm := nodeMapping(ev)
						errs = append(errs, requireKeys(evm, evPath, []string{"type", "ref"})...)
						if n, ok := evm["type"]; ok {
							errs = append(errs, enumCheck(n, evPath+".type", []string{"role", "project", "cert"})...)
						}
					}
				}
			}
		}
	}
	return errs
}

// --- projects.yaml ---
// Top-level key: projects (list)
// Each: id, name, type, period, summary (strings), repo (optional), tech_stack (map), responsibilities, impact, ats_tags

func validateProjects(root *yaml.Node) []string {
	var errs []string
	top := nodeMapping(root)
	errs = append(errs, requireKeys(top, "projects.yaml", []string{"projects"})...)
	projsNode, ok := top["projects"]
	if !ok {
		return errs
	}
	if projsNode.Kind != yaml.SequenceNode {
		errs = append(errs, "projects: expected list")
		return errs
	}
	for i, proj := range projsNode.Content {
		path := fmt.Sprintf("projects[%d]", i)
		if proj.Kind != yaml.MappingNode {
			errs = append(errs, fmt.Sprintf("%s: expected map", path))
			continue
		}
		pm := nodeMapping(proj)
		errs = append(errs, requireKeys(pm, path, []string{"id", "name", "type", "period", "summary", "tech_stack", "responsibilities", "impact", "ats_tags"})...)

		for _, listKey := range []string{"responsibilities", "impact", "ats_tags"} {
			if n, ok := pm[listKey]; ok {
				errs = append(errs, nodeStringList(n, path+"."+listKey)...)
			}
		}

		if tsNode, ok := pm["tech_stack"]; ok {
			if tsNode.Kind != yaml.MappingNode {
				errs = append(errs, fmt.Sprintf("%s.tech_stack: expected map", path))
			} else {
				tsm := nodeMapping(tsNode)
				for _, tsKey := range []string{"languages", "frameworks", "infrastructure", "patterns", "tools"} {
					if n, ok := tsm[tsKey]; ok {
						errs = append(errs, nodeStringList(n, fmt.Sprintf("%s.tech_stack.%s", path, tsKey))...)
					}
				}
			}
		}
	}
	return errs
}

// --- experience.yaml ---
// Top-level keys: roles (list), education (list)
// Roles: id, company, title, period, type (enum), skills_used, bullets, ats_tags
// Education: institution, degree, field, period

func validateExperience(root *yaml.Node) []string {
	var errs []string
	top := nodeMapping(root)
	errs = append(errs, requireKeys(top, "experience.yaml", []string{"roles", "education"})...)

	if rolesNode, ok := top["roles"]; ok {
		if rolesNode.Kind != yaml.SequenceNode {
			errs = append(errs, "roles: expected list")
		} else {
			for i, role := range rolesNode.Content {
				path := fmt.Sprintf("roles[%d]", i)
				if role.Kind != yaml.MappingNode {
					errs = append(errs, fmt.Sprintf("%s: expected map", path))
					continue
				}
				rm := nodeMapping(role)
				errs = append(errs, requireKeys(rm, path, []string{"id", "company", "title", "period", "type", "skills_used", "bullets", "ats_tags"})...)
				if n, ok := rm["type"]; ok {
					errs = append(errs, enumCheck(n, path+".type", []string{"research", "internship"})...)
				}
				for _, listKey := range []string{"skills_used", "bullets", "ats_tags"} {
					if n, ok := rm[listKey]; ok {
						errs = append(errs, nodeStringList(n, path+"."+listKey)...)
					}
				}
			}
		}
	}

	if eduNode, ok := top["education"]; ok {
		if eduNode.Kind != yaml.SequenceNode {
			errs = append(errs, "education: expected list")
		} else {
			for i, edu := range eduNode.Content {
				path := fmt.Sprintf("education[%d]", i)
				if edu.Kind != yaml.MappingNode {
					errs = append(errs, fmt.Sprintf("%s: expected map", path))
					continue
				}
				em := nodeMapping(edu)
				errs = append(errs, requireKeys(em, path, []string{"institution", "degree", "field", "period"})...)
			}
		}
	}

	return errs
}
