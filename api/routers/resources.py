"""
资源管理 API 路由

提供资产、模型、工作流、模板的 CRUD 操作。
"""

from fastapi import APIRouter, HTTPException, UploadFile, File, Query, BackgroundTasks
from fastapi.responses import Response
from typing import List, Optional

from api.models.schemas import (
    AssetInfo, AssetCreate, AssetUpdate, AssetTypeEnum, ResourceCategory,
    ModelInfo, ModelTypeEnum, SavedWorkflow, WorkflowTemplate,
    DirectoryInfo, NodeInstance, NodeConnection
)
from api.services.resource_service import resource_service

router = APIRouter(prefix="/resources", tags=["resources"])


# ==================== 资产管理 ====================

@router.get("/assets", response_model=List[AssetInfo])
async def list_assets(
    category: Optional[ResourceCategory] = None,
    asset_type: Optional[AssetTypeEnum] = None,
    search: Optional[str] = None
) -> List[AssetInfo]:
    """
    列出资产
    
    - **category**: 资源大类 (inputs/outputs/models)
    - **asset_type**: 资产类型 (dataset/spectrum/gds/structure/design/simulation/export)
    - **search**: 搜索关键词
    """
    return await resource_service.list_assets(
        category=category,
        asset_type=asset_type,
        search=search
    )


@router.get("/assets/{asset_id}", response_model=AssetInfo)
async def get_asset(asset_id: str, category: ResourceCategory) -> AssetInfo:
    """获取单个资产详情"""
    asset = await resource_service.get_asset(asset_id, category)
    if not asset:
        raise HTTPException(status_code=404, detail="Asset not found")
    return asset


@router.post("/assets", response_model=AssetInfo)
async def create_asset(
    name: str,
    category: ResourceCategory,
    asset_type: AssetTypeEnum,
    file: UploadFile = File(...),
    description: Optional[str] = None,
    metadata: Optional[str] = None
) -> AssetInfo:
    """
    创建资产（上传文件）
    
    - **name**: 资产名称
    - **category**: 资源大类
    - **asset_type**: 资产类型
    - **file**: 上传的文件
    - **description**: 描述
    - **metadata**: JSON 格式的元数据
    """
    import json
    
    file_content = await file.read()
    meta_dict = {}
    if metadata:
        try:
            meta_dict = json.loads(metadata)
        except:
            pass
    
    return await resource_service.create_asset(
        name=name,
        category=category,
        asset_type=asset_type,
        file_content=file_content,
        description=description,
        metadata=meta_dict
    )


@router.patch("/assets/{asset_id}", response_model=AssetInfo)
async def update_asset(
    asset_id: str,
    category: ResourceCategory,
    update: AssetUpdate
) -> AssetInfo:
    """更新资产元数据"""
    asset = await resource_service.update_asset(
        asset_id=asset_id,
        category=category,
        name=update.name,
        description=update.description,
        metadata=update.metadata
    )
    if not asset:
        raise HTTPException(status_code=404, detail="Asset not found")
    return asset


@router.delete("/assets/{asset_id}")
async def delete_asset(asset_id: str, category: ResourceCategory) -> dict:
    """删除资产"""
    success = await resource_service.delete_asset(asset_id, category)
    if not success:
        raise HTTPException(status_code=404, detail="Asset not found")
    return {"message": "Asset deleted", "id": asset_id}


@router.get("/assets/{asset_id}/download")
async def download_asset(asset_id: str, category: ResourceCategory) -> Response:
    """下载资产文件"""
    content = await resource_service.get_asset_content(asset_id, category)
    if content is None:
        raise HTTPException(status_code=404, detail="Asset not found")
    
    asset = await resource_service.get_asset(asset_id, category)
    filename = asset.name if asset else asset_id
    
    return Response(
        content=content,
        media_type="application/octet-stream",
        headers={
            "Content-Disposition": f'attachment; filename="{filename}"'
        }
    )


# ==================== 模型管理 ====================

@router.get("/models", response_model=List[ModelInfo])
async def list_models(
    model_type: Optional[ModelTypeEnum] = None,
    challenge: Optional[str] = None,
    pretrained_only: bool = False
) -> List[ModelInfo]:
    """
    列出模型
    
    - **model_type**: 模型类型 (tnn/mdn/cgan/pinn/gnn/hilab)
    - **challenge**: 挑战名称 (grating_coupler/metagrating/wavelength_demux)
    - **pretrained_only**: 仅显示预训练模型
    """
    return await resource_service.list_models(
        model_type=model_type,
        challenge=challenge,
        pretrained_only=pretrained_only
    )


@router.get("/models/{model_id}", response_model=ModelInfo)
async def get_model(model_id: str) -> ModelInfo:
    """获取模型详情"""
    model = await resource_service.get_model(model_id)
    if not model:
        raise HTTPException(status_code=404, detail="Model not found")
    return model


@router.delete("/models/{model_id}")
async def delete_model(model_id: str) -> dict:
    """删除模型"""
    success = await resource_service.delete_model(model_id)
    if not success:
        raise HTTPException(status_code=404, detail="Model not found")
    return {"message": "Model deleted", "id": model_id}


@router.get("/models/{model_id}/download")
async def download_model(model_id: str) -> Response:
    """下载模型权重"""
    model = await resource_service.get_model(model_id)
    if not model:
        raise HTTPException(status_code=404, detail="Model not found")
    
    import os
    file_path = os.path.join(str(resource_service.project_root), model.file_path)
    
    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail="Model file not found")
    
    with open(file_path, 'rb') as f:
        content = f.read()
    
    return Response(
        content=content,
        media_type="application/octet-stream",
        headers={
            "Content-Disposition": f'attachment; filename="{model.name}.pt"'
        }
    )


# ==================== 工作流管理 ====================

@router.get("/workflows", response_model=List[SavedWorkflow])
async def list_saved_workflows(search: Optional[str] = None) -> List[SavedWorkflow]:
    """列出已保存的工作流"""
    return await resource_service.list_saved_workflows(search=search)


@router.get("/workflows/{workflow_id}", response_model=SavedWorkflow)
async def get_saved_workflow(workflow_id: str) -> SavedWorkflow:
    """获取已保存的工作流"""
    workflow = await resource_service.get_saved_workflow(workflow_id)
    if not workflow:
        raise HTTPException(status_code=404, detail="Workflow not found")
    return workflow


@router.post("/workflows", response_model=SavedWorkflow)
async def save_workflow(
    name: str,
    nodes: List[NodeInstance],
    edges: List[NodeConnection],
    description: Optional[str] = None,
    tags: Optional[List[str]] = None
) -> SavedWorkflow:
    """
    保存工作流
    
    - **name**: 工作流名称
    - **nodes**: 节点实例列表
    - **edges**: 连接列表
    - **description**: 描述
    - **tags**: 标签
    """
    return await resource_service.save_workflow(
        name=name,
        nodes=nodes,
        edges=edges,
        description=description,
        tags=tags
    )


@router.delete("/workflows/{workflow_id}")
async def delete_saved_workflow(workflow_id: str) -> dict:
    """删除已保存的工作流"""
    success = await resource_service.delete_saved_workflow(workflow_id)
    if not success:
        raise HTTPException(status_code=404, detail="Workflow not found")
    return {"message": "Workflow deleted", "id": workflow_id}


# ==================== 模板管理 ====================

@router.get("/templates", response_model=List[WorkflowTemplate])
async def list_templates(category: Optional[str] = None) -> List[WorkflowTemplate]:
    """列出工作流模板"""
    return await resource_service.list_templates(category=category)


@router.get("/templates/{template_id}", response_model=WorkflowTemplate)
async def get_template(template_id: str) -> WorkflowTemplate:
    """获取模板详情"""
    template = await resource_service.get_template(template_id)
    if not template:
        raise HTTPException(status_code=404, detail="Template not found")
    return template


# ==================== 目录信息 ====================

@router.get("/directories/{category}", response_model=DirectoryInfo)
async def get_directory_info(category: ResourceCategory) -> DirectoryInfo:
    """
    获取目录信息
    
    - **category**: 资源大类 (inputs/outputs/models/workflows)
    """
    return await resource_service.get_directory_info(category)


# ==================== 批量操作 ====================

@router.post("/assets/batch-delete")
async def batch_delete_assets(
    asset_ids: List[str],
    category: ResourceCategory
) -> dict:
    """批量删除资产"""
    deleted = []
    failed = []
    
    for asset_id in asset_ids:
        success = await resource_service.delete_asset(asset_id, category)
        if success:
            deleted.append(asset_id)
        else:
            failed.append(asset_id)
    
    return {
        "deleted": deleted,
        "failed": failed,
        "total": len(asset_ids)
    }
