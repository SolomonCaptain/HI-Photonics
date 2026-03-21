"""
资源管理服务

管理 inputs、outputs、op_models 目录下的各类资源文件。
"""

import os
import json
import uuid
import shutil
from pathlib import Path
from typing import Dict, List, Optional, Any
from datetime import datetime

from api.models.schemas import (
    AssetInfo, AssetCreate, AssetUpdate, AssetTypeEnum, ResourceCategory,
    ModelInfo, ModelTypeEnum, SavedWorkflow, WorkflowTemplate,
    DirectoryInfo, NodeInstance, NodeConnection
)


class ResourceService:
    """资源管理服务"""

    def __init__(self, project_root: Path = None):
        if project_root is None:
            project_root = Path(__file__).parent.parent.parent
        self.project_root = project_root
        self.inputs_dir = project_root / "inputs"
        self.outputs_dir = project_root / "outputs"
        self.models_dir = project_root / "op_models"
        self.workflows_dir = project_root / "workflows"

        # 目录结构配置
        self.category_paths = {
            ResourceCategory.INPUTS: {
                AssetTypeEnum.DATASET: self.inputs_dir / "datasets",
                AssetTypeEnum.SPECTRUM: self.inputs_dir / "spectra",
                AssetTypeEnum.GDS: self.inputs_dir / "gds",
                AssetTypeEnum.STRUCTURE: self.inputs_dir / "structures",
            },
            ResourceCategory.OUTPUTS: {
                AssetTypeEnum.DESIGN: self.outputs_dir / "designs",
                AssetTypeEnum.SIMULATION: self.outputs_dir / "simulations",
                AssetTypeEnum.EXPORT: self.outputs_dir / "exports",
            },
            ResourceCategory.MODELS: {
                AssetTypeEnum.MODEL_WEIGHTS: self.models_dir / "pretrained",
            },
        }

        # 元数据文件名
        self.metadata_filename = ".resource_meta.json"
        
        # 初始化目录
        self._ensure_directories()

    def _ensure_directories(self):
        """确保所有目录存在"""
        for category, type_paths in self.category_paths.items():
            for path in type_paths.values():
                path.mkdir(parents=True, exist_ok=True)
        
        # 确保工作流目录
        (self.workflows_dir / "saved").mkdir(parents=True, exist_ok=True)
        (self.workflows_dir / "templates").mkdir(parents=True, exist_ok=True)

    def _get_path_for_asset(self, category: ResourceCategory, asset_type: AssetTypeEnum) -> Path:
        """获取资产存储路径"""
        if category in self.category_paths:
            if asset_type in self.category_paths[category]:
                return self.category_paths[category][asset_type]
        raise ValueError(f"Unknown asset type: {asset_type} for category: {category}")

    def _generate_id(self) -> str:
        """生成唯一 ID"""
        return str(uuid.uuid4())[:8]

    def _get_file_size(self, path: Path) -> int:
        """获取文件大小"""
        if path.is_file():
            return path.stat().st_size
        return 0

    def _load_metadata(self, dir_path: Path) -> Dict[str, Any]:
        """加载目录元数据"""
        meta_file = dir_path / self.metadata_filename
        if meta_file.exists():
            try:
                with open(meta_file, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except:
                pass
        return {"assets": {}}

    def _save_metadata(self, dir_path: Path, metadata: Dict[str, Any]):
        """保存目录元数据"""
        meta_file = dir_path / self.metadata_filename
        with open(meta_file, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, ensure_ascii=False, indent=2)

    # ==================== 资产管理 ====================

    async def list_assets(
        self,
        category: Optional[ResourceCategory] = None,
        asset_type: Optional[AssetTypeEnum] = None,
        search: Optional[str] = None
    ) -> List[AssetInfo]:
        """列出资产"""
        assets = []

        # 确定要扫描的目录
        dirs_to_scan = []
        if category and asset_type:
            path = self._get_path_for_asset(category, asset_type)
            dirs_to_scan.append((category, path))
        elif category:
            if category in self.category_paths:
                for at, path in self.category_paths[category].items():
                    dirs_to_scan.append((category, path))
        else:
            for cat, type_paths in self.category_paths.items():
                for at, path in type_paths.items():
                    dirs_to_scan.append((cat, path))

        # 扫描目录
        for cat, dir_path in dirs_to_scan:
            if not dir_path.exists():
                continue
            
            metadata = self._load_metadata(dir_path)
            
            for file_path in dir_path.iterdir():
                if file_path.name.startswith('.'):
                    continue
                if file_path.is_file():
                    asset_id = file_path.stem
                    meta = metadata.get("assets", {}).get(asset_id, {})
                    
                    # 推断资产类型
                    inferred_type = self._infer_asset_type(file_path, cat)
                    
                    asset = AssetInfo(
                        id=asset_id,
                        name=meta.get("name", file_path.stem),
                        type=inferred_type,
                        category=cat,
                        description=meta.get("description"),
                        file_path=str(file_path.relative_to(self.project_root)),
                        file_size=self._get_file_size(file_path),
                        created_at=meta.get("created_at", datetime.fromtimestamp(file_path.stat().st_ctime).isoformat()),
                        updated_at=meta.get("updated_at", datetime.fromtimestamp(file_path.stat().st_mtime).isoformat()),
                        metadata=meta.get("metadata"),
                        thumbnail=meta.get("thumbnail")
                    )
                    
                    # 搜索过滤
                    if search and search.lower() not in asset.name.lower():
                        continue
                    
                    assets.append(asset)

        return assets

    def _infer_asset_type(self, file_path: Path, category: ResourceCategory) -> AssetTypeEnum:
        """根据文件扩展名推断资产类型"""
        suffix = file_path.suffix.lower()
        
        type_mapping = {
            '.h5': AssetTypeEnum.DATASET,
            '.hdf5': AssetTypeEnum.DATASET,
            '.npy': AssetTypeEnum.DATASET,
            '.npz': AssetTypeEnum.DATASET,
            '.csv': AssetTypeEnum.DATASET,
            '.gds': AssetTypeEnum.GDS,
            '.gdsii': AssetTypeEnum.GDS,
            '.png': AssetTypeEnum.SPECTRUM,
            '.jpg': AssetTypeEnum.SPECTRUM,
            '.json': AssetTypeEnum.STRUCTURE,
            '.pt': AssetTypeEnum.MODEL_WEIGHTS,
            '.pth': AssetTypeEnum.MODEL_WEIGHTS,
            '.ckpt': AssetTypeEnum.MODEL_WEIGHTS,
        }
        
        if suffix in type_mapping:
            return type_mapping[suffix]
        
        # 根据目录推断
        if category == ResourceCategory.INPUTS:
            return AssetTypeEnum.DATASET
        elif category == ResourceCategory.OUTPUTS:
            return AssetTypeEnum.DESIGN
        elif category == ResourceCategory.MODELS:
            return AssetTypeEnum.MODEL_WEIGHTS
        
        return AssetTypeEnum.DATASET

    async def get_asset(self, asset_id: str, category: ResourceCategory) -> Optional[AssetInfo]:
        """获取单个资产"""
        assets = await self.list_assets(category=category)
        for asset in assets:
            if asset.id == asset_id:
                return asset
        return None

    async def create_asset(
        self,
        name: str,
        category: ResourceCategory,
        asset_type: AssetTypeEnum,
        file_content: bytes,
        description: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> AssetInfo:
        """创建资产"""
        dir_path = self._get_path_for_asset(category, asset_type)
        asset_id = self._generate_id()
        
        # 确定文件扩展名
        ext_mapping = {
            AssetTypeEnum.DATASET: '.h5',
            AssetTypeEnum.SPECTRUM: '.png',
            AssetTypeEnum.GDS: '.gds',
            AssetTypeEnum.STRUCTURE: '.json',
            AssetTypeEnum.MODEL_WEIGHTS: '.pt',
            AssetTypeEnum.DESIGN: '.json',
            AssetTypeEnum.SIMULATION: '.json',
            AssetTypeEnum.EXPORT: '.gds',
        }
        ext = ext_mapping.get(asset_type, '.dat')
        
        file_name = f"{asset_id}{ext}"
        file_path = dir_path / file_name
        
        # 写入文件
        with open(file_path, 'wb') as f:
            f.write(file_content)
        
        # 更新元数据
        dir_metadata = self._load_metadata(dir_path)
        now = datetime.now().isoformat()
        
        asset_meta = {
            "name": name,
            "description": description,
            "created_at": now,
            "updated_at": now,
            "metadata": metadata or {}
        }
        dir_metadata.setdefault("assets", {})[asset_id] = asset_meta
        self._save_metadata(dir_path, dir_metadata)
        
        return AssetInfo(
            id=asset_id,
            name=name,
            type=asset_type,
            category=category,
            description=description,
            file_path=str(file_path.relative_to(self.project_root)),
            file_size=len(file_content),
            created_at=now,
            updated_at=now,
            metadata=metadata
        )

    async def update_asset(
        self,
        asset_id: str,
        category: ResourceCategory,
        name: Optional[str] = None,
        description: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> Optional[AssetInfo]:
        """更新资产元数据"""
        asset = await self.get_asset(asset_id, category)
        if not asset:
            return None
        
        dir_path = Path(self.project_root / asset.file_path).parent
        dir_metadata = self._load_metadata(dir_path)
        
        if asset_id in dir_metadata.get("assets", {}):
            asset_meta = dir_metadata["assets"][asset_id]
            if name:
                asset_meta["name"] = name
            if description is not None:
                asset_meta["description"] = description
            if metadata:
                asset_meta["metadata"] = {**asset_meta.get("metadata", {}), **metadata}
            asset_meta["updated_at"] = datetime.now().isoformat()
            
            self._save_metadata(dir_path, dir_metadata)
            return await self.get_asset(asset_id, category)
        
        return None

    async def delete_asset(self, asset_id: str, category: ResourceCategory) -> bool:
        """删除资产"""
        asset = await self.get_asset(asset_id, category)
        if not asset:
            return False
        
        file_path = Path(self.project_root / asset.file_path)
        
        # 删除文件
        if file_path.exists():
            file_path.unlink()
        
        # 更新元数据
        dir_path = file_path.parent
        dir_metadata = self._load_metadata(dir_path)
        if asset_id in dir_metadata.get("assets", {}):
            del dir_metadata["assets"][asset_id]
            self._save_metadata(dir_path, dir_metadata)
        
        return True

    async def get_asset_content(self, asset_id: str, category: ResourceCategory) -> Optional[bytes]:
        """获取资产内容"""
        asset = await self.get_asset(asset_id, category)
        if not asset:
            return None
        
        file_path = Path(self.project_root / asset.file_path)
        if file_path.exists():
            with open(file_path, 'rb') as f:
                return f.read()
        return None

    # ==================== 模型管理 ====================

    async def list_models(
        self,
        model_type: Optional[ModelTypeEnum] = None,
        challenge: Optional[str] = None,
        pretrained_only: bool = False
    ) -> List[ModelInfo]:
        """列出模型"""
        models = []
        
        # 扫描预训练模型目录
        pretrained_dir = self.models_dir / "pretrained"
        if pretrained_dir.exists():
            models.extend(self._scan_model_dir(pretrained_dir, is_pretrained=True))
        
        # 扫描自定义模型目录
        custom_dir = self.models_dir / "custom"
        if custom_dir.exists():
            models.extend(self._scan_model_dir(custom_dir, is_pretrained=False))
        
        # 过滤
        if model_type:
            models = [m for m in models if m.type == model_type.value]
        if challenge:
            models = [m for m in models if m.challenge == challenge]
        if pretrained_only:
            models = [m for m in models if m.is_pretrained]
        
        return models

    def _scan_model_dir(self, dir_path: Path, is_pretrained: bool) -> List[ModelInfo]:
        """扫描模型目录"""
        models = []
        meta_file = dir_path / ".models_meta.json"
        
        all_meta = {}
        if meta_file.exists():
            try:
                with open(meta_file, 'r', encoding='utf-8') as f:
                    all_meta = json.load(f)
            except:
                pass
        
        for file_path in dir_path.iterdir():
            if file_path.suffix.lower() not in ['.pt', '.pth', '.ckpt']:
                continue
            
            model_id = file_path.stem
            meta = all_meta.get(model_id, {})
            
            model = ModelInfo(
                id=model_id,
                name=meta.get("name", file_path.stem),
                type=meta.get("type", "hilab"),
                challenge=meta.get("challenge", "grating_coupler"),
                description=meta.get("description"),
                file_path=str(file_path.relative_to(self.project_root)),
                file_size=self._get_file_size(file_path),
                metrics=meta.get("metrics"),
                created_at=meta.get("created_at", datetime.fromtimestamp(file_path.stat().st_ctime).isoformat()),
                updated_at=meta.get("updated_at", datetime.fromtimestamp(file_path.stat().st_mtime).isoformat()),
                is_pretrained=is_pretrained
            )
            models.append(model)
        
        return models

    async def get_model(self, model_id: str) -> Optional[ModelInfo]:
        """获取模型信息"""
        models = await self.list_models()
        for model in models:
            if model.id == model_id:
                return model
        return None

    async def delete_model(self, model_id: str) -> bool:
        """删除模型"""
        model = await self.get_model(model_id)
        if not model:
            return False
        
        file_path = Path(self.project_root / model.file_path)
        if file_path.exists():
            file_path.unlink()
        
        # 更新元数据
        meta_file = file_path.parent / ".models_meta.json"
        if meta_file.exists():
            try:
                with open(meta_file, 'r', encoding='utf-8') as f:
                    all_meta = json.load(f)
                if model_id in all_meta:
                    del all_meta[model_id]
                with open(meta_file, 'w', encoding='utf-8') as f:
                    json.dump(all_meta, f)
            except:
                pass
        
        return True

    # ==================== 工作流管理 ====================

    async def list_saved_workflows(self, search: Optional[str] = None) -> List[SavedWorkflow]:
        """列出已保存的工作流"""
        workflows = []
        saved_dir = self.workflows_dir / "saved"
        
        if not saved_dir.exists():
            return workflows
        
        for file_path in saved_dir.glob("*.json"):
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                
                workflow = SavedWorkflow(
                    id=data.get("id", file_path.stem),
                    name=data.get("name", file_path.stem),
                    description=data.get("description"),
                    nodes=[NodeInstance(**n) for n in data.get("nodes", [])],
                    edges=[NodeConnection(**e) for e in data.get("edges", [])],
                    file_path=str(file_path.relative_to(self.project_root)),
                    created_at=data.get("created_at", ""),
                    updated_at=data.get("updated_at", ""),
                    tags=data.get("tags")
                )
                
                if search and search.lower() not in workflow.name.lower():
                    continue
                
                workflows.append(workflow)
            except Exception as e:
                print(f"Error loading workflow {file_path}: {e}")
        
        return workflows

    async def get_saved_workflow(self, workflow_id: str) -> Optional[SavedWorkflow]:
        """获取已保存的工作流"""
        workflows = await self.list_saved_workflows()
        for workflow in workflows:
            if workflow.id == workflow_id:
                return workflow
        return None

    async def save_workflow(
        self,
        name: str,
        nodes: List[NodeInstance],
        edges: List[NodeConnection],
        description: Optional[str] = None,
        tags: Optional[List[str]] = None
    ) -> SavedWorkflow:
        """保存工作流"""
        saved_dir = self.workflows_dir / "saved"
        saved_dir.mkdir(parents=True, exist_ok=True)
        
        workflow_id = self._generate_id()
        now = datetime.now().isoformat()
        
        data = {
            "id": workflow_id,
            "name": name,
            "description": description,
            "nodes": [n.dict() for n in nodes],
            "edges": [e.dict() for e in edges],
            "created_at": now,
            "updated_at": now,
            "tags": tags or []
        }
        
        file_path = saved_dir / f"{workflow_id}.json"
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        
        return SavedWorkflow(
            id=workflow_id,
            name=name,
            description=description,
            nodes=nodes,
            edges=edges,
            file_path=str(file_path.relative_to(self.project_root)),
            created_at=now,
            updated_at=now,
            tags=tags
        )

    async def delete_saved_workflow(self, workflow_id: str) -> bool:
        """删除已保存的工作流"""
        workflow = await self.get_saved_workflow(workflow_id)
        if not workflow:
            return False
        
        file_path = Path(self.project_root / workflow.file_path)
        if file_path.exists():
            file_path.unlink()
        return True

    # ==================== 模板管理 ====================

    async def list_templates(self, category: Optional[str] = None) -> List[WorkflowTemplate]:
        """列出工作流模板"""
        templates = []
        templates_dir = self.workflows_dir / "templates"
        
        if not templates_dir.exists():
            return templates
        
        for file_path in templates_dir.glob("*.json"):
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                
                template = WorkflowTemplate(
                    id=data.get("id", file_path.stem),
                    name=data.get("name", file_path.stem),
                    description=data.get("description"),
                    category=data.get("category", "general"),
                    icon=data.get("icon", "AccountTree"),
                    nodes=[NodeInstance(**n) for n in data.get("nodes", [])],
                    edges=[NodeConnection(**e) for e in data.get("edges", [])],
                    tags=data.get("tags"),
                    file_path=str(file_path.relative_to(self.project_root))
                )
                
                if category and template.category != category:
                    continue
                
                templates.append(template)
            except Exception as e:
                print(f"Error loading template {file_path}: {e}")
        
        return templates

    async def get_template(self, template_id: str) -> Optional[WorkflowTemplate]:
        """获取模板"""
        templates = await self.list_templates()
        for template in templates:
            if template.id == template_id:
                return template
        return None

    # ==================== 目录信息 ====================

    async def get_directory_info(self, category: ResourceCategory) -> DirectoryInfo:
        """获取目录信息"""
        if category == ResourceCategory.INPUTS:
            base_path = self.inputs_dir
            subdirs = ["datasets", "spectra", "gds", "structures"]
        elif category == ResourceCategory.OUTPUTS:
            base_path = self.outputs_dir
            subdirs = ["designs", "simulations", "exports"]
        elif category == ResourceCategory.MODELS:
            base_path = self.models_dir
            subdirs = ["pretrained", "custom"]
        else:
            base_path = self.workflows_dir
            subdirs = ["saved", "templates"]
        
        total_size = 0
        file_count = 0
        
        if base_path.exists():
            for file_path in base_path.rglob("*"):
                if file_path.is_file() and not file_path.name.startswith('.'):
                    total_size += file_path.stat().st_size
                    file_count += 1
        
        return DirectoryInfo(
            path=str(base_path),
            name=category.value,
            category=category,
            total_size=total_size,
            file_count=file_count,
            subdirectories=subdirs
        )


# 全局服务实例
resource_service = ResourceService()
